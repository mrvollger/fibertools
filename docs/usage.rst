=====
Usage
=====

To use fibertools in a project::

    import fibertools as ft
    from dataclasses import dataclass

    @dataclass
    class Args:
        """Class for keeping track of input data form fibertools"""
        msp_bed12: str = "../tests/data/test_fiber_calls.bed.gz"
        m6a_bed12: str= "../tests/data/test_fiber_m6A_calls.bed.gz"
        genome: str="../tests/data/test.fa.gz"
        dhs: str="../tests/data/dhs.with.null.bed.gz"
        bin_num: int=9
        bin_width: int=40
        model: str=None 
        ...
    # read in fake input    
    args = Args()

    fiberdata = ft.Fiberdata(
        args.msp_bed12,
        args.m6a_bed12
    )
    AT_genome = ft.make_AT_genome(args.genome, fiberdata.both)
    dhs = read_in_bed_file(args.dhs)
    
    fiberdata.make_msp_features(
        AT_genome, bin_num=args.bin_num, bin_width=args.bin_width
    )
    fiberdata.make_percolator_input(dhs, min_tp_msp_len=args.min_tp_msp_len)

    if args.model is None:
        fiberdata.train_accessibility_model(args.out)
    else:
        fiberdata.predict_accessibility(args.model)
        fiberdata.accessibility.to_csv(args.out, sep="\t", index=False)