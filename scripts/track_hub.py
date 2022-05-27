import os
import sys

HUBDIR = "trackHub"

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
os.makedirs(f"{HUBDIR}/", exist_ok=True)

open(f"{HUBDIR}/hub.txt", "w").write(hub)
open(f"{HUBDIR}/genomes.txt", "w").write(genomes.format(ref="hg38"))
trackDb = open(f"{HUBDIR}/trackDb.txt", "w")
trackDb.write(track_comp)
for i in range(75):
    trackDb.write(sub_comp_track.format(i=i + 1))
trackDb.close()
