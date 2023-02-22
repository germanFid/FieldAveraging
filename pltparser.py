import numpy as np

def ParsePlt(path, delimiter, skipHeader):
    data = []
    data.append(np.genfromtxt(path, delimiter='\t', skip_header=skipHeader))

    return data