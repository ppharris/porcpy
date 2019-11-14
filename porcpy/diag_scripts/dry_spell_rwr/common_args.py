#!/usr/bin/env python

from __future__ import print_function, division

from argparse import ArgumentParser
from os.path import isdir


def get_common_args(require_input_dir=False):
    parser = ArgumentParser()

    if require_input_dir:
        parser.add_argument("-i", "--input-dir", dest="dir_regress_in",
                            default=None, required=require_input_dir,
                            help="Directory containing other regression test "
                            "output that are inputs files to this script.")

    parser.add_argument("-o", "--output-dir", dest="dir_regress_out",
                        default=None, required=True,
                        help="Directory for the regression test output files.")

    args = parser.parse_args()

    if require_input_dir and not isdir(args.dir_regress_in):
        raise ValueError("Input dir does not exist: %s" %
                         args.dir_regress_in)

    if not isdir(args.dir_regress_out):
        raise ValueError("Output dir does not exist: %s" %
                         args.dir_regress_out)

    return args


if __name__ == "__main__":
    pass
