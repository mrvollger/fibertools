"""Console script for fibertools."""
#!/usr/bin/env python3
import argparse
from email import header
import sys
import logging
import pickle
import pandas as pd
from fibertools.readutils import read_in_bed_file
from .utils import join_msp_and_m6a
from .classify import make_msp_features
import fibertools as ft
import mokapot
import numpy as np
from sklearn.model_selection import GridSearchCV
import gzip


def make_msp_features_parser(subparsers):
    parser = subparsers.add_parser("features", help="Make MSP features")
    parser.add_argument("msp_bed12", help="MSP bed12 file.")
    parser.add_argument("m6a_bed12", help="m6a bed12 file.")
    parser.add_argument(
        "genome",
        help="Indexed fasta file to read in sequence context. Should have the same chromosomes as referenced in the first column of the two bed files.",
    )
    parser.add_argument("-d", "--dhs", help="dhs", default=None)
    parser.add_argument(
        "-b", "--bin_width", help="size of bin width to use", type=int, default=40
    )
    parser.add_argument(
        "--bin-num",
        help="Number of feature bins to work. Must be an odd number.",
        type=int,
        default=7,
    )
    parser.add_argument("-o", "--out", help="Out file to write to", default=sys.stdout)
    parser.add_argument(
        "-n",
        "--n-rows",
        help="For debugging only reads in n rows.",
        type=int,
        default=None,
    )


def make_bam2bed_parser(subparsers):
    parser = subparsers.add_parser(
        "bam2bed", help="Extract m6a calls from bam and output bed12."
    )


def make_add_m6a_parser(subparsers):
    parser = subparsers.add_parser("add-m6a", help="Make MSP features")
    parser.add_argument("bam", help="ccs bam file.")
    parser.add_argument("m6a", help="m6a bed12 file.")
    parser.add_argument(
        "-o", "--out", help="file to write output bam to.", default=sys.stdout
    )


def make_bed_split_parser(subparsers):
    parser = subparsers.add_parser("split", help="Split a bed over many output files.")
    parser.add_argument("bed", help="A bed file")
    parser.add_argument(
        "-o", "--out-files", help="files to split input across", nargs="+"
    )


def make_accessibility_model_parser(subparsers):
    parser = subparsers.add_parser("model", help="Make MSP features")
    parser.add_argument("msp_bed12", help="MSP bed12 file.")
    parser.add_argument("m6a_bed12", help="m6a bed12 file.")
    parser.add_argument(
        "genome",
        help="Indexed fasta file to read in sequence context. Should have the same chromosomes as referenced in the first column of the two bed files.",
    )
    parser.add_argument("-d", "--dhs", help="dhs", default=None)
    parser.add_argument(
        "-b", "--bin_width", help="size of bin width to use", type=int, default=40
    )
    parser.add_argument(
        "--bin-num",
        help="Number of feature bins to work. Must be an odd number.",
        type=int,
        default=9,
    )
    parser.add_argument(
        "--min-tp-msp-len",
        help="min msp length to be in the DHS positive training set.",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--spacer-size",
        help="adjust minimum distance between fibers for them to be in the same bin.",
        type=int,
        default=100,
    )
    parser.add_argument("--train-fdr", type=float, default=0.10)
    parser.add_argument("--test-fdr", type=float, default=0.05)
    parser.add_argument("-m", "--model", default=None)
    parser.add_argument(
        "-o", "--out", help="Write the accessibility model to this file."
    )
    parser.add_argument(
        "-n",
        "--n-rows",
        help="For debugging only reads in n rows.",
        type=int,
        default=None,
    )


def make_model_input(args):
    df = join_msp_and_m6a(args)
    AT_genome = ft.make_AT_genome(args.genome, df)
    df["bin"] = ft.utils.disjoint_bins(df.st, df.en, spacer_size=args.spacer_size)
    msp_feat = make_msp_features(args, df, AT_genome)
    dhs = read_in_bed_file(args.dhs, n_rows=args.n_rows)
    pin = ft.classify.make_percolator_input(
        msp_feat, dhs, min_tp_msp_len=args.min_tp_msp_len
    )
    return (df, msp_feat, pin)


def train_accessibility_model(args):
    _df, _msp_feat, pin = make_model_input(args)
    # with open("indata.dat", "wb") as f:
    #    pickle.dump(pd.DataFrame(df.to_dicts()), f)
    #    pickle.dump(pin, f)
    moka_conf, models = ft.classify.make_accessibility_model(
        pin, train_fdr=args.train_fdr, test_fdr=args.test_fdr
    )

    with open(args.out, "wb") as f:
        pickle.dump(moka_conf.psms, f)
        pickle.dump(models, f)


def apply_accessibility_model(args):
    moka_conf_psms, models = list(ft.utils.load_all(args.model))
    df, msp_feat, pin = make_model_input(args)
    # need ot make a pandas
    df = pd.DataFrame(df.to_dicts())
    test_psms = mokapot.read_pin(pin)
    all_scores = [model.predict(test_psms) for model in models]
    # scores = np.mean(np.array(all_scores), axis=0)
    # scores = np.amin(np.array(all_scores), axis=0)
    scores = np.amax(np.array(all_scores), axis=0)

    q_values = ft.classify.find_nearest_q_values(
        moka_conf_psms["mokapot score"], moka_conf_psms["mokapot q-value"], scores
    )
    merged = pin.copy()
    merged["mokapot score"] = scores
    merged["mokapot q-value"] = q_values

    out = msp_feat
    out["qValue"] = 1
    out["strand"] = "+"
    out.loc[merged.SpecId, "qValue"] = merged["mokapot q-value"]
    out["tst"] = out["st"]
    out["ten"] = out["en"]
    out["color"] = "147,112,219"
    # out.loc[ out.qValue < 0.5, "color"] =  "255,255,0"
    # out.loc[ out.qValue < 0.4, "color"] =  "255,255,0"
    out.loc[out.qValue < 0.3, "color"] = "255,255,0"
    out.loc[out.qValue < 0.2, "color"] = "255,140,0"
    out.loc[out.qValue < 0.1, "color"] = "255,0,0"
    out.loc[out.qValue < 0.05, "color"] = "139,0,0"
    out = out.merge(df[["fiber", "bin"]])

    df["spacer_st"] = df.apply(
        lambda row: (row["bst_msp"] + row["bsize_msp"])[:-1] + row["st"], axis=1
    )
    df["spacer_en"] = df.apply(lambda row: row["bst_msp"][1:] + row["st"] + 1, axis=1)
    z = df.explode(["spacer_st", "spacer_en"])
    z["qValue"] = 1
    z["strand"] = "+"
    z["tst"] = z["spacer_st"]
    z["ten"] = z["spacer_st"]  # make non msp thin blocks
    z["color"] = "230,230,230"
    z.drop(columns=["st", "en"], inplace=True)
    z.rename(columns={"spacer_st": "st", "spacer_en": "en"}, inplace=True)

    out_cols = [
        "ct",
        "st",
        "en",
        "fiber",
        "score",
        "strand",
        "tst",
        "ten",
        "color",
        "qValue",
        "bin",
    ]
    final_out = (
        pd.concat([out[out_cols], z[out_cols]])
        .sort_values(["ct", "st"])
        .rename(columns={"ct": "#ct"})
    )
    final_out["score"] = (final_out.qValue * 100).astype(int)
    final_out.to_csv(args.out, sep="\t", index=False)


def split_bed_over_files(args):
    bed = ft.read_in_bed_file(args.bed)
    logging.debug("Read in bed file.")
    index_splits = np.array_split(np.arange(bed.shape[0]), len(args.out_files))
    for index, out_file in zip(index_splits, args.out_files):
        if out_file.endswith(".gz"):
            with gzip.open(out_file, "wb") as f:
                bed[index].to_csv(f, sep="\t", has_header=False)
        else:
            bed[index].to_csv(out_file, sep="\t", has_header=False)


def parse():
    """Console script for fibertools."""
    parser = argparse.ArgumentParser(
        description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available subcommand for fibertools", required=True
    )
    make_msp_features_parser(subparsers)
    make_bam2bed_parser(subparsers)
    make_add_m6a_parser(subparsers)
    make_accessibility_model_parser(subparsers)
    make_bed_split_parser(subparsers)
    # shared arguments
    parser.add_argument("-t", "--threads", help="n threads to use", type=int, default=1)
    parser.add_argument(
        "-v", "--verbose", help="increase logging verbosity", action="store_true"
    )
    args = parser.parse_args()

    log_format = "[%(levelname)s][Time elapsed (ms) %(relativeCreated)d]: %(message)s"
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(format=log_format, level=log_level)

    if args.command == "features":
        # run steps
        df = join_msp_and_m6a(args)
        AT_genome = ft.make_AT_genome(args.genome, df)
        out = make_msp_features(args, df, AT_genome)
        dhs = read_in_bed_file(args.dhs, n_rows=args.n_rows)
        ft.utils.make_percolator_input(out, dhs)

        logging.debug(f"Sorting and writing features to out.")
        (
            out.sort_values(by=["ct", "st"])
            .rename(columns={"ct": "#ct"})
            .to_csv(args.out, index=False, sep="\t", header=True)
        )
    elif args.command == "add_m6a":
        _m6a = ft.read_in_bed12_file(args.m6a)
    elif args.command == "model":
        if args.model is None:
            train_accessibility_model(args)
        else:
            apply_accessibility_model(args)
    elif args.command == "split":
        split_bed_over_files(args)

    return 0
