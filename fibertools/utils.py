"""Utilities for reads and making features out of m6a and MSPs.
"""
import pandas as pd
from numba import njit
import numpy as np
import logging
import sys
import fibertools as ft

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


@njit
def get_msp_mid(st, msp_st, msp_size):
    return st + msp_st + msp_size // 2


@njit
def get_bin_AT(bin_starts, is_at, bin_width=40):
    """_summary_
    Args:
        bin_starts (_type_): start sites of bins
        is_at (bool): array of AT sites
        bin_width (int, optional):  Defaults to 40.

    Returns:
        list of how many AT positions are in each bin.
    """
    rtn = []  # np.zeros(len(bins))
    for bin_st in bin_starts:
        bin_en = bin_st + bin_width
        if bin_st < 0 and bin_en < 0:
            rtn.append(0)
            continue
        if bin_st < 0:
            bin_st = 0
        if bin_en > len(is_at):
            bin_en = len(is_at)
        if bin_en <= bin_st:
            rtn.append(0)
            continue
        rtn.append(is_at[bin_st:bin_en].sum())
    return np.array(rtn)


@njit
def get_bin_m6a(bin_starts, bst_m6a, bin_width=40):
    rtn = []  # np.zeros(len(bins))
    for bin_st in bin_starts:
        bin_en = bin_st + bin_width
        count = 0
        for m6a in bst_m6a:
            if m6a >= bin_st and m6a < bin_en:
                count += 1
        rtn.append(count)
    return np.array(rtn)


def make_bin_starts(mid, bin_width=40, bin_num=5):
    """_summary_

    Args:
        mid (int): np array with the middle positions of an MSPs.
        bin_width (int, optional): _description_. Defaults to 40.
        bin_num (int, optional): _description_. Defaults to 5.

    Returns:
        Returns a 2d array of MSP bin start positions.
    """
    start = mid - bin_width * (bin_num // 2 + 1) + bin_width // 2
    rtn = []
    for i in range(bin_num):
        rtn.append(start + bin_width * i)
    return np.stack(rtn, axis=1)


def get_msp_features(row, AT_genome, bin_width=40, bin_num=5):
    is_at = AT_genome[row["ct"]][row["st"] : row["en"]]
    typed_bst_m6a = row["bst_m6a"]
    rtn = []
    mids = get_msp_mid(0, row["bst_msp"], row["bsize_msp"])
    all_bin_starts = make_bin_starts(mids, bin_width=bin_width, bin_num=bin_num)
    for msp_st, msp_size, bin_starts in zip(
        row["bst_msp"], row["bsize_msp"], all_bin_starts
    ):
        msp_en = msp_st + msp_size
        if msp_en <= msp_st:
            continue
        m6a_counts = get_bin_m6a(bin_starts, typed_bst_m6a, bin_width=bin_width)
        AT_count = get_bin_AT(bin_starts, is_at, bin_width=bin_width)
        msp_AT = is_at[msp_st:msp_en].sum()
        msp_m6a = ((typed_bst_m6a >= msp_st) & (typed_bst_m6a < msp_en)).sum()
        rtn.append(
            {
                "ct": row["ct"],
                "st": row["st"] + msp_st,
                "en": row["st"] + msp_st + msp_size,
                "fiber": row["fiber"],
                "msp_m6a": msp_m6a,
                "msp_AT": msp_AT,
                "m6a_count": m6a_counts,
                "AT_count": AT_count,
            }
        )
    return rtn


def split_to_ints(df, col, sep=","):
    """Split a columns with list of ints seperated by
    "sep" into a numpy array quickly.

    Args:
        df (dataframe): dataframe that is like a bed12 file.
        col (str): column name within the dataframe to split up.
        sep (str, optional): Defaults to ",".

    Returns:
        column: New column that is a list of numpy array of ints.
    """
    return df[col].apply(lambda x: np.fromstring(x, sep=sep, dtype=np.int32))


def join_msp_and_m6a(args):
    msp = ft.read_in_bed12_file(args.msp_bed12, n_rows=args.n_rows, tag="msp")
    logging.debug("Read in MSP file.")
    m6a = ft.read_in_bed12_file(args.m6a_bed12, n_rows=args.n_rows, tag="m6a")
    logging.debug("Read in m6a file.")
    both = m6a.join(msp, on=["ct", "st", "en", "fiber"]).drop(
        ["ten_msp", "tst_msp", "ten_m6a", "tst_m6a", "bsize_m6a"]
    )
    logging.debug("Joined m6a and MSP data.")
    return both


def make_msp_features(args, df, AT_genome):
    msp_stuff = []
    rows = df.to_dicts()
    for idx, row in enumerate(rows):
        if row["bsize_msp"] is None or row["bst_msp"] is None:
            continue
        if logging.DEBUG >= logging.root.level:
            sys.stderr.write(
                f"\r[DEBUG]: Added featrues to {(idx+1)/len(rows):.3%} of MSPs."
            )
        msp_stuff += get_msp_features(
            row, AT_genome, bin_width=args.bin_width, bin_num=args.bin_num
        )
    logging.debug("")
    logging.debug("Expanding bed12s into individual MSPs.")

    z = pd.DataFrame(msp_stuff)
    # for some reason the MSPs sometimes have negative lengths
    # z = z[(z["st"] < z["en"])]
    # Make more MSP featrues columns
    z["bin_m6a_frac"] = z.m6a_count / z.AT_count
    z["m6a_frac"] = z.msp_m6a / z.msp_AT
    z["msp_len"] = z.en - z.st
    m6a_fracs = pd.DataFrame(
        z["bin_m6a_frac"].tolist(),
        columns=[f"m6a_frac_{i}" for i in range(args.bin_num)],
    )
    m6a_counts = pd.DataFrame(
        z["m6a_count"].tolist(), columns=[f"m6a_count_{i}" for i in range(args.bin_num)]
    )
    AT_counts = pd.DataFrame(
        z["AT_count"].tolist(), columns=[f"AT_count_{i}" for i in range(args.bin_num)]
    )
    out = (
        pd.concat([z, m6a_fracs, m6a_counts, AT_counts], axis=1)
        .drop(["bin_m6a_frac", "m6a_count", "AT_count"], axis=1)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    # out = out.astype(
    #    {
    #        "st": "int",
    #        "en": "int",
    #        "msp_len": "int",
    #    }
    # )
    return out
