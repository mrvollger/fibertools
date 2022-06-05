import os
import sys
from .utils import disjoint_bins
import pandas as pd


def generate_trackhub(df, trackhub_dir="trackHub", ref="hg38", spacer_size=100):
    hub = """
hub fiberseq
shortLabel fiberseq
longLabel fiberseq
genomesFile genomes.txt
email mvollger.edu
    """

    genomes = """
genome {ref}
trackDb trackDb.txt
    """

    track_comp = """
track fiberseq
compositeTrack on
shortLabel fiberseq
longLabel fiberseq
type bigBed 9 +
visibility dense
maxItems 100000
maxHeightPixels 200:200:1
    """
    sub_comp_track = """
    track bin{i}
    parent fiberseq
    bigDataUrl bins/bin.{i}.bed.bb
    shortLabel bin{i}
    longLabel bin{i}
    priority {i}
    type bigBed 9 +
    itemRgb on
    visibility dense
    maxHeightPixels 1:1:1
    
    """
    os.makedirs(f"{trackhub_dir}/", exist_ok=True)

    open(f"{trackhub_dir}/hub.txt", "w").write(hub)
    open(f"{trackhub_dir}/genomes.txt", "w").write(genomes.format(ref=ref))
    trackDb = open(f"{trackhub_dir}/trackDb.txt", "w")
    trackDb.write(track_comp)
    for i in range(75):
        trackDb.write(sub_comp_track.format(i=i + 1))
    trackDb.close()

    # write the bins to file
    os.makedirs(f"{trackhub_dir}/bed", exist_ok=True)
    os.makedirs(f"{trackhub_dir}/bins", exist_ok=True)
    df["bin"] = disjoint_bins(
        df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], spacer_size=spacer_size
    )
    for cur_bin in sorted(df.bin.unique()):
        sys.stderr.write(f"\r{cur_bin}")
        (
            df.loc[df.bin == cur_bin].to_csv(
                f"{trackhub_dir}/bed/bin.{cur_bin}.bed.gz",
                sep="\t",
                index=False,
                compression="gzip",
            )
        )
