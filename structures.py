import pandas as pd
import io


class StreamData:
    """Field information class"""
    dataset = pd.DataFrame()
    i, j = 0, 0

    def __init__(self, csvstr: str, i: int, j: int):
        # TODO: Check for correct data it should be csv-string
        self.dataset = pd.read_csv(io.StringIO(csvstr))
        self.i = i
        self.j = j

    def get_dataset_line(self, i: int, j: int):
        """Returns DataFrame (dataset) line"""
        row = i + j * self.j
        return self.dataset.iloc[[row]]


def parse_plt(path: str) -> StreamData:
    """Parses .plt file

    Args:
        path (str): path to plt file

    Returns:
        StreamData: parsed StreamData
    """

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

        for k in range(i * j):
            s = file.readline()
            s = s.replace("\t", ',')
            csvstr += s

    return StreamData(csvstr, i, j)


def output_plt(data: StreamData, original_file: str, new_file: str, header=2):
    """Outputs StreamData to file

    Args:
        data (StreamData): Actual StreamData to output
        original_file (str): Path to Original .plt file program got data from
        new_file (str): Path to New .plt file
        header (int, optional): Number of header strings. Defaults to 2.
    """

    with open(original_file, "r") as fo:
        with open(new_file, "w") as fn:

            for i in range(header):
                line = fo.readline()
                fn.write(line)

            for i in data.dataset.values:
                flag = True
                for j in i:
                    if not flag:
                        fn.write("\t")
                    else:
                        flag = False

                    fn.write(str(j))
                fn.write("\n")


def advance_to_vxu(data: StreamData):
    """Outputs 2d list of Vx/U

    Returns:
        2dList
    """

    w, h = data.i, data.j
    output = [[float(0) for x in range(h)] for y in range(w)]

    for i in range(w):
        for j in range(h):
            output[i][j] = data.get_dataset_line(j, i)["Vx/U"].item()

    return output


def advance_to_column(data: StreamData, column_name: str):
    """Outputs 2d list of column setting"""

    column = data.dataset[column_name]

    # Get the number of rows in the column
    num_rows = column.shape[0]

    # Create a list to hold the data
    data_list = []

    # Iterate over the rows of the column and append each value to the list
    for i in range(num_rows):
        data_list.append(column[i])

    # Reshape the list into a 2D list with x and y dimensions
    x_dim = data.i
    y_dim = data.j

    data_2d = [data_list[i:i + x_dim] for i in range(0, y_dim * x_dim, x_dim)]

    # Return the result
    return data_2d


def update_dataset_column(data: StreamData, column: str, list):
    """Updates dataset column of StreamData

    Args:
        data (StreamData): StreamData to update
        column (str): Name of column to update
        list (_type_): 2d list with updated values
    """

    n = 0
    for i in range(data.i):
        for j in range(data.j):
            data.dataset.iloc[n][column] = list[i][j]
            n += 1
