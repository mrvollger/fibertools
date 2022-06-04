import pysam
from pybedtools import BedTool
import numpy as np
from tqdm import tqdm


def coordinateConversion_MMTag(sequence, base, modification_coords):
    """Sequence is array of bases. Base is a base
    for conversion to the coordinate system
    used in the MM tag."""

    mask = sequence == bytes(base, encoding="utf-8")
    # find all masks = boolean Array

    coords = modification_coords[
        sequence[modification_coords] == bytes(base, encoding="utf-8")
    ]
    # when working with double stranded data we can only use modifications
    # on the specifc base that the modification would fall on on that strand
    # i.e. As  on + , Ts on -, we only want the mods that = that base of interest

    MM_coords = ",".join(list((np.diff(np.cumsum(mask)[coords]) - 1).astype(str)))

    return MM_coords


def make_m6a_tagged_bam(bam, m6a_df, output_bam):
    """_summary_

    Example:
        bam = pysam.AlignmentFile(sys.argv[1], "rb")
        bed = BedTool(sys.argv[2])
        output_bam_name = sys.argv[1][:-3] + "m6A.tagged.bam"
        header = bam.header.to_dict()
        output_bam = pysam.AlignmentFile(output_bam_name, "wb", header=header)
        make_m6a_tragged_bam(bam, bed, output_bam)

    Args:
        bam (_type_): _description_
        m6a_df (_type_): _description_
        output_bam (_type_): _description_
    """
    m6a_dict = dict(zip(m6a_df.fiber, zip(m6a_df.bst, m6a_df.st)))

    with output_bam as out_f:  # open output bam file
        with bam as in_f:  # open input bam file, iterate through,
            # and write modification to output bam
            for read in tqdm(in_f):
                name_list = read.query_name.split("/")
                new_name = str(name_list[1] + "/" + name_list[0])

                # new_name = str(read.query_name)
                if new_name in m6a_dict:
                    ap = np.vstack(read.get_aligned_pairs(matches_only=True)).T

                    # adjust bed to genomic coordinates
                    # search for adjusted coordinates
                    # in ap -- take the molecular coordinates that aligned
                    genomic_m6a = m6a_dict[new_name][0] + m6a_dict[new_name][1]

                    mol_m6a = ap[
                        0, np.isin(ap[1], genomic_m6a)
                    ]  # this is coordinates of query sequence
                    # we need coordinates of query alignemnt sequence

                    sequence = np.frombuffer(
                        bytes(read.query_sequence, "utf-8"), dtype="S1"
                    )

                    ##########
                    # np.sum(np.isin(sequence[mol_m6a[1:-1]],['G','C']))
                    # -- this will retrieve count of methylation calls with poor alignment
                    # where reference does not match up A/T
                    #########

                    # # need to encode per strand
                    # # A+a encodes forward strand ( check As , offset of As )
                    # # T-a encodes reverse strand ( check Ts , offset of Ts )

                    # need to grab all A and all T positions
                    # and then frame methylation positions in
                    # the appropriate base-space

                    A_mods = coordinateConversion_MMTag(sequence, "A", mol_m6a)
                    T_mods = coordinateConversion_MMTag(sequence, "T", mol_m6a)
                    mods = "A+a," + A_mods + ";" + "T-a," + T_mods + ";"
                    read.tags += [("MM:Z:", mods)]

                out_f.write(read)
