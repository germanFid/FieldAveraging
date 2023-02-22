import numpy as np
import argparse
import pltparser

parser = argparse.ArgumentParser()
parser.add_argument('inputfile', help='input plt file')
parser.add_argument('--Header', '-H', help='number of header lines', type=int)

args = parser.parse_args()

DEFAULT_HEADER = 2

if args.Header:
    DEFAULT_HEADER = args.Header

data = pltparser.ParsePlt(args.inputfile, '\t', DEFAULT_HEADER)

print(data)
