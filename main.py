import argparse
import structures
import logging
import averager

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('inputfile', help='input csv file', type=str)

# TODO: add x, y, z args

parser.add_argument('--lines', '-l', help='number of header lines', type=int)

parser.add_argument('--verbose', '-v', help='verbose progress', action='store_true')

parser.add_argument('--vv', help='verbose MORE progress', action='store_true')

parser.add_argument('--leave', help='tkinter-output mode', action='store_true')

parser.add_argument('--dimensions', '-d', help='dimensions seperated by comma', type=str)

parser.add_argument('--job', '-j', help='job to do with opened Data',
                    choices=['basic2d', 'basic2d_paral', 'basic3d', 'basic3d_paral', 'gauss'],
                    type=str)

parser.add_argument('--columns', '-c', help='columns to do jobs seperated by comma', type=str)

parser.add_argument('--radius', '-r', help='averaging radius', type=int)
parser.add_argument('--iterations', '-i', help='number of iterations', type=int)

parser.add_argument('--outfile', '-o', help='output file', type=str)

args = parser.parse_args()

DEFAULT_HEADER = 2
DEFAULT_VERBOSE = False
DEFAULT_MORE_VERBOSE = False
DEFAULT_LEAVE = True
DEFAULT_RADIUS = 1
DEFAULT_ITERATIONS = 1
DIM_X, DIM_Y, DIM_Z = 0, 0, 0

LOG_FORMAT = '\n> %(asctime)s %(message)s\n'
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger()


def do_job(job: str, data: structures.StreamData, columns, iters, radius, verbose=False):
    if iters != 1:
        vv = True

    else:
        vv = False

    result = None

    if job == 'basic2d':
        for col in columns:
            print("\n")
            logger.warning("Performing Job on " + col)
            result = averager.basic_2d_averaging_iterations(
                np.asarray(structures.advance_to_column(data, col)), iters, radius, 1, False, verbose)
            print()

    elif job == 'basic2d_paral':
        for col in columns:
            print("\n")
            logger.warning("Performing Job on " + col)
            result = averager.basic_2d_averaging_iterations(
                np.asarray(structures.advance_to_column(data, col)), iters, radius, 4, verbose, DEFAULT_MORE_VERBOSE, DEFAULT_LEAVE)

    elif job == 'basic3d':
        for col in columns:
            print("\n")
            logger.warning("Performing Job on " + col)
            result = averager.basic_3d_averaging_iterations(
                np.asarray(structures.advance_to_column(data, col)), iters, radius, 1, False, verbose)

    elif job == 'basic3d_paral':
        for col in columns:
            print("")
            logger.warning("Performing Job on " + col)
            result = averager.basic_2d_averaging_iterations(
                np.asarray(structures.advance_to_column(data, col)), iters, radius, 4, verbose, DEFAULT_MORE_VERBOSE, DEFAULT_LEAVE)

    elif job == 'gauss':
        pass


if args.lines:
    DEFAULT_HEADER = args.Header

if args.verbose:
    DEFAULT_VERBOSE = True

if args.vv:
    DEFAULT_MORE_VERBOSE = True

if args.dimensions:
    dims = str(args.dimensions).split(',')
    if len(dims) not in [2, 3]:
        logger.error("Incorrect number of dimensions!")
        exit()

    else:
        DIM_X, DIM_Y = int(dims[0]), int(dims[1])

        if len(dims) == 3:
            DIM_Z = int(dims[2])

if args.radius:
    DEFAULT_RADIUS = args.radius

if args.iterations:
    DEFAULT_ITERATIONS = args.iterations

if args.leave:
    DEFAULT_LEAVE = False

if __name__ == '__main__':
    with open('.logo.txt') as file:
        print(file.read())

    logger.warning('Averager Init Done!')

    data = structures.StreamData(args.inputfile, DIM_X, DIM_Y, DIM_Z)
    logger.warning('File Loading Done!')

    if DEFAULT_VERBOSE:
        print(data.dataset)

    if args.job:
        columns = str(args.columns).split(",")

        for col in columns:  # first check if column exists
            if col not in data.dataset:
                logger.error("Wrong Column: " + col)
                exit(1)

        logger.warning("Started Job: " + args.job)
        do_job(args.job, data, columns, DEFAULT_ITERATIONS, DEFAULT_RADIUS, DEFAULT_VERBOSE)

        logger.warning('All jobs Done!')
    print()