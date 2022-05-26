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
        ...
    
    args = Args()
    df = ft.join_msp_and_m6a(args)
    AT_genome = ft.make_AT_genome(args.genome, df)  
    msp_with_features = make_msp_features(args, df, AT_genome)


