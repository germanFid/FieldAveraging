import numpy as np
import pandas as pd
import io

class StreamData:
    dataset = pd.DataFrame().empty
    i, j = 0, 0

    def __init__(self, csvstr: str, i: int, j: int):
        # TODO: Check for correct data it should be csv-string
        self.dataset = pd.read_csv(io.StringIO(csvstr))
        self.i = i
        self.j = j

def parse_plt(path):
    csvstr = ''
    i, j = 0, 0
    with open(path, "r") as file:
        first = file.readline()
        first = first.replace(',', ' ')
        first = first.replace('"', ' ')

        vars = first.split()[2:]
        for v in vars:
            csvstr += v + ','

        csvstr = csvstr[:-1]
        csvstr += '\n'

        second = file.readline()
        second = second.replace('=', ' ')
        second = second.replace(',', ' ')

        ij = second.split()
        i = int(ij[2])
        j = int(ij[4])

        for i in range(i * j):
            s = file.readline()
            s = s.replace("\t", ',')
            csvstr += s
    
    return StreamData(csvstr, i, j)
