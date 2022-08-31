#!/usr/bin/python

import argparse
import sys
import time

import QDMpy
from QDMpy.core.qdm import QDM


def main(argv):
    tstart = time.process_time()

    parser = argparse.ArgumentParser(
        description="Calculate the B111 field from ODMR data recorded with QDMio made QDM"
    )
    parser.add_argument(
        "-i",
        "--input",
        help="input path, location of the QDM data files and LED/laser images.",
        required=True,
    )
    parser.add_argument(
        "-b",
        "--binfactor",
        type=int,
        help="Binning factor of the ODMR data. Default: 1",
        default=1,
        required=False,
    )
    parser.add_argument(
        "-d",
        "--diamond",
        type=str,
        help="Type of diamond used in the experiment. Default: 'auto'",
        default=None,
        required=False,
    )
    parser.add_argument(
        "-gf",
        "--globalfluorescence",
        type=float,
        help="Global fluorescence of the sample. Default: 0.2",
        default=0.2,
        required=False,
    )
    parser.add_argument(
        "--debug",
        help="sets logging to DEBUG level",
        action="store_true",
        default=False,
        required=False,
    )

    args = parser.parse_args()

    if args.debug:
        QDMpy.LOG.setLevel("DEBUG")
    else:
        QDMpy.LOG.setLevel("INFO")

    qdm_obj = QDM.from_qdmio(args.input, diamond_type=args.diamond)
    qdm_obj.bin_data(bin_factor=args.binfactor)
    qdm_obj.correct_glob_fluorescecne(glob_fluorescence=args.globalfluorescence)
    qdm_obj.fit_ODMR()
    qdm_obj.export_qdmio()
    QDMpy.LOG.info(f"QDMpy finished in {time.process_time() - tstart:.2f} seconds")


if __name__ == "__main__":
    main(sys.argv[1:])
