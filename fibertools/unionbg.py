import pyd4
import tempfile
import numpy as np
from numba import njit
import logging
import fibertools as ft
import polars as pl


@njit
def chrom_bg(sts, ens, chrom_len):
    chrom = np.zeros(chrom_len, dtype=np.uint16)
    to_add = np.uint64(1)
    for st, en in zip(sts, ens):
        chrom[st:en] += to_add
    return chrom


def df_to_bg(df, genome):
    bg_data = {}
    for g in df.groupby("ct"):
        chrom = g.ct[0]
        bg_data[chrom] = chrom_bg(g.st.to_numpy(), g.en.to_numpy(), genome[chrom])
    return bg_data


def make_d4_from_df(df, genome, d4_f):
    chroms = list(zip(genome.keys(), genome.values()))
    writer = pyd4.D4Builder(d4_f).add_chroms(chroms).for_sparse_data().get_writer()

    for chrom, data in df_to_bg(df, genome).items():
        writer.write_np_array(chrom, 0, data)
    writer.close()


def make_temp_d4_from_df(df, genome):
    temp = tempfile.NamedTemporaryFile(suffix=".d4")
    make_d4_from_df(df, genome, temp.name)
    return temp


def make_union_d4_from_df(df, genome, group_col, d4_f):
    out_files = []
    for idx, g in enumerate(df.groupby([group_col])):
        g_n = g[group_col][0]
        out_files.append((g_n, make_temp_d4_from_df(g, genome)))
        logging.debug(f"Made d4 for group: {g_n}")

    merged = pyd4.D4Merger(d4_f)
    for tag, d4 in sorted(out_files):
        merged.add_tagged_track(str(tag), d4.name)
    merged.merge()
    # close files
    [d4.close() for _tag, d4 in out_files]


def bed2d4(args):
    df = ft.read_in_bed_file(args.bed)
    if args.column == "score":
        df = df.with_column(
            pl.when(pl.col("column_9") == "230,230,230")
            .then(101)
            .otherwise(pl.col("column_5"))
            .alias(args.column)
        )
    genome = {line.split()[0]: int(line.split()[1]) for line in open(args.genome)}
    make_union_d4_from_df(df, genome, args.column, args.d4)
