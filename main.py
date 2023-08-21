import argparse
import structures
import logging
import averager
import useful_graphics

import numpy as np

from numba import cuda
from numba.cuda import list_devices
import os


def detect_cuda():
    """
    Detect supported CUDA hardware and print a summary of the detected hardware.

    Returns a boolean indicating whether any supported devices were detected.
    """
    devlist = list_devices()
    print('Found %d CUDA devices' % len(devlist))
    supported_count = 0
    for dev in devlist:
        attrs = []
        cc = dev.compute_capability
        kernel_timeout = dev.KERNEL_EXEC_TIMEOUT
        tcc = dev.TCC_DRIVER
        fp32_to_fp64_ratio = dev.SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO
        attrs += [('Compute Capability', '%d.%d' % cc)]
        attrs += [('PCI Device ID', dev.PCI_DEVICE_ID)]
        attrs += [('PCI Bus ID', dev.PCI_BUS_ID)]
        attrs += [('UUID', dev.uuid)]
        attrs += [('Watchdog', 'Enabled' if kernel_timeout else 'Disabled')]
        if os.name == "nt":
            attrs += [('Compute Mode', 'TCC' if tcc else 'WDDM')]
        attrs += [('FP32/FP64 Performance Ratio', fp32_to_fp64_ratio)]
        if cc < (3, 5):
            support = '[NOT SUPPORTED: CC < 3.5]'
        elif cc < (5, 0):
            support = '[SUPPORTED (DEPRECATED)]'
            supported_count += 1
        else:
            support = '[SUPPORTED]'
            supported_count += 1

        print('id %d    %20s %40s' % (dev.id, dev.name, support))
        for key, val in attrs:
            print('%40s: %s' % (key, val))

    print('Summary:')
    print('\t%d/%d devices are supported' % (supported_count, len(devlist)))
    return supported_count


number_of_devices = detect_cuda()
match number_of_devices:
    case 0:
        "No CUDA devices were found, loading without CUDA code."
    case 1:
        import averager_cuda
    case _:
        import averager_cuda
        try:
            inputed_id = int(input("Select CUDA device id:"))
            cuda.select_device(inputed_id)
        except Exception:
            print("Inputed id is greater than number of devices.")
            successfull = False
            while (not successfull):
                inputed_id = int(input("Try another one to be between 0 and %d:"
                                       % (number_of_devices - 1)))
                successfull = inputed_id >= 0 and inputed_id <= (number_of_devices - 1)
                print("Error: %d is not between 0 and %d..."
                      % (inputed_id, (number_of_devices - 1)))
            cuda.select_device(inputed_id)
        else:
            print("Successfully selected id %d cuda device!" % (inputed_id))


parser = argparse.ArgumentParser()
parser.add_argument('inputfile', help='input csv file', type=str)

# TODO: add x, y, z args

parser.add_argument('--lines', '-l', help='number of header lines', type=int)

parser.add_argument('--verbose', '-v', help='verbose progress', action='store_true')

parser.add_argument('--vv', help='verbose MORE progress', action='store_true')

parser.add_argument('--leave', help='tkinter-output mode', action='store_true')

parser.add_argument('--dimensions', '-d', help='dimensions seperated by comma', type=str)

parser.add_argument('--job', '-j',
                    help='job to do with opened Data {' + 'basic2d' +
                    "'basic2d_paral', 'basic3d', 'basic3d_paral', 'gauss', 'plot2d', 'scatter3d'}",
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

job_types = {
    "basic_2d": averager.basic_2d_averaging_iterations,
    "basic_2d_paral": averager.basic_2d_averaging_iterations,
    "basic_3d":  averager.basic_3d_averaging_iterations,
    "basic_3d_paral": averager.basic_3d_averaging_iterations
}

graphics_types = ['plot2d', 'scatter_3d']


def perform(func, data: structures.StreamData, columns, iters, radius,
            verbose=False, _job=""):
    rs = []
    proc = 0

    if _job.find("paral") != -1:
        proc = 4

    else:
        proc = 1

    for col in columns:
        print()
        logger.warning(f"Performing Job {_job} on " + col)
        rs.append({"result": func(
            np.asarray(structures.advance_to_column(data, col)), iters, radius, proc, verbose,
            DEFAULT_MORE_VERBOSE, DEFAULT_LEAVE), "column": col})

    return rs


def do_job(jobs, data: structures.StreamData, columns, iters, radius, verbose=False):
    results = {
        "jobs_avgs": [],
        "jobs_grpx": []
    }

    for job in jobs:
        if job in graphics_types:
            continue

        results['jobs_avgs'].append(perform(job_types[job], data,
                                            columns, iters, radius, verbose, job))

    if len(results['jobs_avgs']) > 0:
        for elem in results['jobs_avgs'][0]:
            structures.update_dataset_column(data, elem["column"], elem["result"])

    if 'plot2d' in jobs:
        for col in columns:
            print()
            logger.warning("Performing Job plot2d on " + col)
            useful_graphics.plot_2d(structures.advance_to_column(data, col), title=col)
        useful_graphics.plt.show()

    if 'scatter3d' in jobs:
        for col in columns:
            print()
            logger.warning("Performing Job scatter3d on " + col)
            useful_graphics.scatter_3d_array(structures.advance_to_column(data, col), title=col)
        useful_graphics.plt.show()

    return results


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
        jobs = str(args.job).split(",")

        for col in columns:  # first check if column exists
            if col not in data.dataset:
                logger.error("Wrong Column: " + col)
                exit(1)

        logger.warning("Started Job: " + args.job)
        results = do_job(jobs, data, columns, DEFAULT_ITERATIONS, DEFAULT_RADIUS, DEFAULT_VERBOSE)

        logger.warning('All jobs Done!')

        if args.outfile:
            if len(results['jobs_avgs']) > 1:
                logger.warning("Cannot Perform Saving on Multiple jobs!")
                exit(0)

            try:
                structures.save_temp_streamdata(data, args.outfile)
                logger.warning("Saving to " + args.outfile + ".out.csv done!")

            except Exception as ex:
                print(ex)
                logger.error("Error saving with your filename! Saving as out.csv")
                structures.save_temp_streamdata(data, "out.csv")

            print()
    print()
