import argparse
import structures

parser = argparse.ArgumentParser()
parser.add_argument('inputfile', help='input plt file')
parser.add_argument('--Header', '-H', help='number of header lines', type=int)

args = parser.parse_args()

DEFAULT_HEADER = 2

if args.Header:
    DEFAULT_HEADER = args.Header

data = structures.parse_plt(args.inputfile)
print(data.dataset.to_string())

# averager.test()

# result = averager.FieldAveraging(data)

# pprint(data)
# pprint(result)
