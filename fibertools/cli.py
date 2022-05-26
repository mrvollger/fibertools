"""Console script for fibertools."""
#!/usr/bin/env python3
import argparse
import sys
import logging

from fibertools.readutils import read_in_bed_file
from .utils import make_msp_features, join_msp_and_m6a
import fibertools as ft


def make_msp_featrues_parser(subparsers):
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


def parse():
    """Console script for fibertools."""
    parser = argparse.ArgumentParser(
        description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available subcommand for fibertools", required=True
    )
    make_msp_featrues_parser(subparsers)
    make_bam2bed_parser(subparsers)
    make_add_m6a_parser(subparsers)
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

        logging.debug(f"Sorting and writting features to out.")
        (
            out.sort_values(by=["ct", "st"])
            .rename(columns={"ct": "#ct"})
            .to_csv(args.out, index=False, sep="\t", header=True)
        )
    elif args.command == "add_m6a":
        m6a = ft.read_in_bed12_file(args.m6a)

    return 0
