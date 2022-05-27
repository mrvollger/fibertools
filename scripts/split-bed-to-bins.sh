#!/usr/bin/env bash

HUBDIR="trackHub"
mkdir -p $HUBDIR/bins
BED="browser.bed.gz"
G="hg38.chrom.sizes"

ls tmp/bin*bed.gz |
    parallel -n 1 \
        "zcat {} | cut -f 1-9 > {}.tmp && bedToBigBed -type=bed9 {}.tmp $G $HUBDIR/bins/{/.}.bb; rm {}.tmp"

exit
zcat $BED | grep -v "^#" | cut -f 10 | sort | uniq |
    parallel -n 1 \
        "zcat $BED | grep -w {}$ | cut -f 1-9 > tmp.{}.tmp &&  bedToBigBed -type=bed9 tmp.{}.tmp $G $HUBDIR/bins/bin{}.bb && rm tmp.{}.tmp"
