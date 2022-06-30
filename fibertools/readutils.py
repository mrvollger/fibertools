from .utils import split_to_ints
import logging
import pysam
import polars as pl
import numpy as np
import sys


def read_in_bed12_file(bed_file, n_rows=None, tag=None, trim=True):
    """Read a bed12 file into a polars dataframe.

    Args:
        bed_file (string): path to bed12 file.
        n_rows (int, optional): only read the first n rows. Defaults to None.
        tag (string, optional): Adds a string the end of the columns names. Defaults to None.
        trim (bool, optional): Trim the first and last block of the bed12.

    Returns:
        pl.DataFrame: Dataframe of bed12 file.
    """
    col_names = [
        "ct",
        "st",
        "en",
        "fiber",
        "score",
        "strand",
        "tst",
        "ten",
        "color",
        "bct",
        "bsize",
        "bst",
    ]
    df = pl.read_csv(
        bed_file,
        sep="\t",
        new_columns=col_names,
        has_header=False,
        n_rows=n_rows,
    )
    df["bst"] = split_to_ints(df, "bst", trim=trim)
    df["bsize"] = split_to_ints(df, "bsize", trim=trim)
    if tag is not None:
        df.columns = [
            f"{col}_{tag}" if idx > 4 else col for idx, col in enumerate(df.columns)
        ]
    return df


def make_AT_genome(genome_file, df):
    """_summary_

    Args:
        genome_file (string): Path to fasta file.
        df (pandas): Dataframe with "ct" column.

    Returns:
        A dictionary of boolean numpy arrays
        indicating at each base whether it is AT.
    """
    genome = {}
    for rec in pysam.FastxFile(genome_file):
        genome[rec.name] = rec.sequence.upper()

    records_in_data = df.ct.unique()
    # takes about 7 minutes for 3 GB genome
    AT_genome = {}
    for rec in genome:
        if rec not in records_in_data:
            continue

        if logging.DEBUG >= logging.root.level:
            sys.stderr.write(f"\r[DEBUG]: Processing {rec} from genome.")

        # tmp = np.array(list(genome[rec]))
        # AT_genome[rec] = (tmp == "A") | (tmp == "T")
        # new faster version?
        tmp_arr = np.frombuffer(bytes(genome[rec], "utf-8"), dtype="S1")
        AT_genome[rec] = (tmp_arr == b"T") | (tmp_arr == b"A")

    logging.debug("")
    return AT_genome


def read_in_bed_file(bed_file, n_rows=None, tag=None):
    """Read a bed file into a polars dataframe.

    Args:
        bed_file (string): path to bed file.
        n_rows (int, optional): only read the first n rows. Defaults to None.
        tag (string, optional): Adds a string the end of the columns names. Defaults to None.

    Returns:
        pl.DataFrame: Dataframe of bed12 file.
    """
    col_names = [
        "ct",
        "st",
        "en",
        "fiber",
        "score",
        "strand",
        "tst",
        "ten",
        "color",
        "bct",
        "bsize",
        "bst",
    ]
    df = pl.read_csv(
        bed_file,
        sep="\t",
        comment_char="#",
        has_header=False,
        n_rows=n_rows,
        # encoding="utf8-lossy",
        # ignore_errors = True,
        quote_char=None,
        low_memory=True,
        use_pyarrow=True,
    )
    # df["bst"] = split_to_ints(df, "bst")
    # df["bsize"] = split_to_ints(df, "bsize")

    logging.debug(df.columns)
    if tag is not None:
        df.columns = [
            f"{col}_{tag}" if idx > 4 else col for idx, col in enumerate(df.columns)
        ]
    first_four = ["ct", "st", "en", "name"]
    df.columns = [
        first_four[idx] if idx < 4 else col for idx, col in enumerate(df.columns)
    ]
    logging.debug(df.columns)
    return df
