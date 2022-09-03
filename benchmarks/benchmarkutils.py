"""
Utilities for creating 'benchmark_*' scripts
"""

import argparse
import cProfile
import timeit

def setup_argument_parser():
    """
    Return a `ArgumentParser` with benchmarking argument
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', action='store_true', default=False)
    return parser

def benchmark(statement, profile_fname=None, profile=False, number=1, globals=None):
    """
    Benchmark a statement
    """
    if profile_fname is None:
        profile_fname = 'temp.profile'

    if profile:
        cProfile.run(statement, profile_fname)
    else:
        time = timeit.timeit(statement, globals=globals, number=number)
        print(f"Runtime: {time:.2e} s")
