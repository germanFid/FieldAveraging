import multiprocessing
import numpy as np
from tqdm import tqdm


def average_this_3d_point(i: int, j: int, k: int, in_field: np.ndarray, radius: int) -> float:
    """Basic method of 3-Dimensional averaging. Takes average value of
    all point around given point with given radius.

    Args:
        i (int): index in row
        j (int): index in column
        k (int): index in depth
        in_field (np.ndarray): field to get average value from
        radius (int): averaging radius around this point

    Returns:
        float: peasantly averaged value of our 3d point in field
    """
    n = len(in_field)
    m = len(in_field[0])
    d = len(in_field[0][0])
    i_start = max(0, i - radius)
    i_end = min(n - 1, i + radius)
    j_start = max(0, j - radius)
    j_end = min(m - 1, j + radius)
    k_start = max(0, k - radius)
    k_end = min(d - 1, k + radius)
    window_size = (i_end - i_start + 1) * (j_end - j_start + 1) * (k_end - k_start + 1)
    window_sum = np.sum(in_field[i_start:i_end + 1, j_start:j_end + 1, k_start:k_end + 1])
    return window_sum / window_size


def basic_3d_array_averaging(inputed_field: np.ndarray, radius: int,
                             visuals: bool = False) -> np.ndarray:
    """Function takes field and use basic 3d averaging method. Gives back averaged field

    Args:
        inputed_field (np.ndarray): field to get averaged
        radius (int): averaging radius around this point

    Returns:
        np.ndarray: peasantly averaged 3d field
    """
    n = len(inputed_field)
    m = len(inputed_field[0])
    d = len(inputed_field[0][0])
    output_field = np.zeros((n, m, d))
    if visuals:
        with tqdm(total=n * m * d) as pbar:
            for i in range(n):
                for j in range(m):
                    for k in range(d):
                        output_field[i][j][k] = average_this_3d_point(i, j, k, inputed_field,
                                                                      radius)
                        pbar.update(1)
    else:
        for i in range(n):
            for j in range(m):
                for k in range(d):
                    output_field[i][j][k] = average_this_3d_point(i, j, k, inputed_field, radius)
    return output_field


def process_func_3d(args):
    i, j, k, inputed_field, radius, average_this_3d_point_func = args
    return average_this_3d_point_func(i, j, k, inputed_field, radius)


def basic_3d_array_averaging_parallel(inputed_field: np.ndarray,
                                      radius: int, max_processes: int = 4,
                                      visuals: bool = False) -> np.ndarray:
    """Basic method of 3-Dimensional averaging using parallel computations.
    Takes average value of all point around given point with given radius.

    Args:
        inputed_field (NDArray): field to get averaged
        radius (int): averaging radius around this point
        max_processes (int): maximum of processes to use
        visuals (bool): enables progress bar verbose

    Returns:
        NDArray: peasantly averaged 2d field
    """
    n, m, d = inputed_field.shape
    output_field = np.zeros((n, m))
    pool = multiprocessing.Pool(processes=max_processes)
    args_list = [(i, j, k, inputed_field, radius, average_this_2d_point)
                 for i in range(n) for j in range(m) for k in range(d)]
    chunksize = int(max([1, (n * m * d) / (4 * max_processes)]))
    if visuals:
        results = list(tqdm(pool.imap(process_func_3d, args_list, chunksize=chunksize),
                            total=(n * m * d), miniters=1000))
    else:
        results = list(pool.imap(process_func_3d, args_list, chunksize=chunksize))
    pool.close()
    f = 0
    for i in range(0, n):
        for j in range(0, m):
            for k in range(0, d):
                output_field[i][j] = results[f]
                f += 1

    return output_field


def average_this_2d_point(i: int, j: int, in_field: np.ndarray, radius: int) -> float:
    n, m = in_field.shape
    i_start = max(0, i - radius)
    i_end = min(n - 1, i + radius)
    j_start = max(0, j - radius)
    j_end = min(m - 1, j + radius)
    window_size = (i_end - i_start + 1) * (j_end - j_start + 1)
    window_sum = np.sum(in_field[i_start:i_end + 1, j_start:j_end + 1])

    return window_sum / window_size


def basic_2d_array_averaging(inputed_field: np.ndarray, radius: int,
                             visuals: bool = False) -> np.ndarray:
    """Basic method of 2-Dimensional averaging. Takes average value of
    all point around given point with given radius.

    Args:
        inputed_field (NDArray): field to get averaged
        radius (int): averaging radius around this point
        visuals (bool): enables progress bar verbose

    Returns:
        NDArray: peasantly averaged 2d field
    """
    n, m = inputed_field.shape
    output_field = np.zeros((n, m))
    if visuals:
        with tqdm(total=n * m) as pbar:
            for i in range(n):
                for j in range(m):
                    output_field[i][j] = average_this_2d_point(i, j, inputed_field, radius)
                    pbar.update(1)
    else:
        for i in range(n):
            for j in range(m):
                output_field[i][j] = average_this_2d_point(i, j, inputed_field, radius)
    return output_field


def process_func_2d(args):
    i, j, inputed_field, radius, average_this_2d_point_func = args
    return average_this_2d_point_func(i, j, inputed_field, radius)


def basic_2d_array_averaging_parallel(inputed_field: np.ndarray,
                                      radius: int, max_processes: int = 4,
                                      visuals: bool = False) -> np.ndarray:
    """Basic method of 2-Dimensional averaging using parallel computations.
    Takes average value of all point around given point with given radius.

    Args:
        inputed_field (NDArray): field to get averaged
        radius (int): averaging radius around this point
        max_processes (int): maximum of processes to use
        visuals (bool): enables progress bar verbose

    Returns:
        NDArray: peasantly averaged 2d field
    """
    n, m = inputed_field.shape
    output_field = np.zeros((n, m))
    pool = multiprocessing.Pool(processes=max_processes)
    args_list = [(i, j, inputed_field, radius, average_this_2d_point)
                 for i in range(n) for j in range(m)]
    chunksize = int(max([1, (n * m) / (4 * max_processes)]))
    if visuals:
        results = list(tqdm(pool.imap(process_func_2d, args_list, chunksize=chunksize),
                            total=(n * m), miniters=1000))
    else:
        results = list(pool.imap(process_func_2d, args_list, chunksize=chunksize))
    pool.close()
    k = 0
    for i in range(0, n):
        for j in range(0, m):
            output_field[i][j] = results[k]
            k += 1

    return output_field


def test():
    averaging_width = 1
    w, h = 5, 3
    input_field = [[float(0) for y in range(h)] for x in range(w)]
    input_field[4][2] = 15
    print(input_field)
    print(basic_2d_array_averaging(input_field, averaging_width))
