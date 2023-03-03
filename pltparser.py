import numpy as np


def parse_plt(path, delimiter, skip_header):
    data = []
    data.append(np.genfromtxt(path, delimiter=delimiter,
                              skip_header=skip_header))

    return data
