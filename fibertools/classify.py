"""Utilities for reads and making features out of m6a and MSPs.
"""
import pandas as pd
from numba import njit
import numpy as np
import fibertools as ft
import mokapot
import logging
import sys
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


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
                "score": row["score"],
                "fiber_m6a_count": row["bct_m6a"],
                "fiber_AT_count": is_at.sum(),
                "fiber_m6a_frac": row["bct_m6a"] / is_at.sum(),
                "msp_m6a": msp_m6a,
                "msp_AT": msp_AT,
                "m6a_count": m6a_counts,
                "AT_count": AT_count,
            }
        )
    return rtn


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
    return out


def make_percolator_input(msp_features_df, dhs_df, sort=True, min_tp_msp_len=40):
    """write a input file that works with percolator.

    Args:
        msp_features_df (_type_): _description_
        out_file (_type_): _description_

    return None
    """
    # need to add:   SpecId	Label ... Peptide Proteins
    dhs_null = ft.utils.n_overlaps(msp_features_df, dhs_df[dhs_df.name != "DHS"])
    dhs_true = ft.utils.n_overlaps(msp_features_df, dhs_df[dhs_df.name == "DHS"])

    out_df = msp_features_df.copy()

    out_df.insert(1, "Label", 0)
    out_df.loc[dhs_true > 0, "Label"] = 1
    out_df.loc[dhs_null > 0, "Label"] = -1
    # if the msp is short make it null
    out_df.loc[(dhs_true > 0) & (out_df.msp_len <= min_tp_msp_len), "Label"] = -1

    condition = (dhs_null > 0) | (dhs_true > 0)

    if "SpecId" in out_df.columns:
        out_df.drop("SpecId", axis=1, inplace=True)
    out_df.insert(0, "SpecId", out_df.index)

    to_remove = ["ct", "st", "en", "fiber"]
    out_df.drop(to_remove, axis=1, inplace=True)

    out_df["Peptide"] = out_df.SpecId
    out_df["Proteins"] = out_df.SpecId
    out_df["scannr"] = out_df.SpecId
    out_df["log_msp_len"] = np.log(out_df["msp_len"])

    if sort:
        out_df.sort_values(["Label"], ascending=False, inplace=True)
    return out_df


def find_nearest_q_values(orig_scores, orig_q_values, new_scores):
    orig_scores = np.flip(orig_scores)
    orig_q_values = np.flip(orig_q_values)
    idxs = np.searchsorted(orig_scores, new_scores, side="left")
    idxs[idxs >= len(orig_q_values)] = len(orig_q_values) - 1
    return np.array(orig_q_values)[idxs]


def train_classifier(train, subset_max_train=200_000, test_fdr=0.05, train_fdr=0.1):
    min_size = 1
    train_psms = mokapot.read_pin(train[train.msp_len >= min_size])
    scale_pos_weight = sum(train.Label == -1) / sum(train.Label == 1)
    grid = {
        "n_estimators": [50, 100, 150],
        "scale_pos_weight": [scale_pos_weight],  # [0.5, 1, 2], #np.logspace(0, 2, 3),
        "max_depth": [3, 6, 9],
        "min_child_weight": [3, 6, 9, 12],
        "gamma": [0.1, 1, 10],
    }
    xgb_mod = GridSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric="auc"),
        param_grid=grid,
        cv=3,
        scoring="roc_auc",
        verbose=2,
    )
    mod = mokapot.Model(xgb_mod, train_fdr=train_fdr, subset_max_train=subset_max_train)
    mokapot_conf, models = mokapot.brew(train_psms, mod, test_fdr=test_fdr)
    return (mokapot_conf, models)


def assign_classifier_fdr(pin_data, models, mokapot_conf):
    psms = mokapot.read_pin(pin_data)
    all_scores = [model.predict(pin_data) for model in models]
    scores = np.mean(np.array(all_scores), axis=0)
    # scores = np.amin(np.array(all_scores), axis=0)
    # scores = np.amax(np.array(all_scores), axis=0)

    q_values = find_nearest_q_values(
        mokapot_conf.psms["mokapot score"], mokapot_conf.psms["mokapot q-value"], scores
    )
    return q_values


def make_accessibility_model(
    pin, train_fdr=0.10, test_fdr=0.05, subset_max_train=500_000
):
    logging.debug(f"dataset size: {pin.shape}")
    logging.debug(f"dataset label counts: {pin.Label.value_counts()}")
    train = pin[(pin.Label != 0)].copy()
    logging.debug(f"train size: {train.shape}")
    min_size = 1

    train_psms = mokapot.read_pin(train[train.msp_len >= min_size])
    scale_pos_weight = sum(train.Label == -1) / sum(train.Label == 1)
    grid = {
        "n_estimators": [25, 50, 100],
        "scale_pos_weight": [scale_pos_weight],  # [0.5, 1, 2], #np.logspace(0, 2, 3),
        "max_depth": [3, 6, 9],
        "min_child_weight": [3, 6, 9],
        "gamma": [0.1, 1, 10],
    }
    xgb_mod = GridSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric="auc"),
        param_grid=grid,
        cv=3,
        scoring="roc_auc",
        verbose=2,
    )
    mod = mokapot.Model(xgb_mod, train_fdr=train_fdr, subset_max_train=subset_max_train)
    moka_conf, models = mokapot.brew(train_psms, mod, test_fdr=test_fdr)
    return (moka_conf, models)
