import multiprocessing
import numpy as np
from tqdm import tqdm
from typing import Tuple
from numba import njit


@njit
def average_this_3d_point(i: int, j: int, k: int, in_field: np.ndarray, radius: int) -> float:
    """
    Basic method of 3-Dimensional averaging. Takes average value of
    all point around given point with given radius.
    Args:
        i (int): index in row
        j (int): index in column
        k (int): index in depth
        in_field (np.ndarray): field to get average value from
        radius (int): averaging radius around array point
    Returns:
        float: peasantly averaged value of our 3d point in field
    """
    n, m, d = in_field.shape
    i_start = max(0, i - radius)
    i_end = min(n - 1, i + radius)
    j_start = max(0, j - radius)
    j_end = min(m - 1, j + radius)
    k_start = max(0, k - radius)
    k_end = min(d - 1, k + radius)
    window_size = (i_end - i_start + 1) * \
        (j_end - j_start + 1) * (k_end - k_start + 1)
    window_sum = np.sum(
        in_field[i_start:i_end + 1, j_start:j_end + 1, k_start:k_end + 1])
    return window_sum / window_size


def basic_3d_array_averaging(inputed_field: np.ndarray, radius: int,
                             visuals: bool = False) -> np.ndarray:
    """
    Function takes field and use basic 3d averaging method. Gives back averaged field
    Args:
        inputed_field (np.ndarray): field to get averaged
        radius (int): averaging radius around array point
    Returns:
        np.ndarray: peasantly averaged 3d field
    """
    n, m, d = inputed_field.shape
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
                    output_field[i][j][k] = average_this_3d_point(
                        i, j, k, inputed_field, radius)
    return output_field


def process_func_3d(args):
    i, j, k, inputed_field, radius, average_this_3d_point_func = args
    return average_this_3d_point_func(i, j, k, inputed_field, radius)


def basic_3d_array_averaging_parallel(inputed_field: np.ndarray,
                                      radius: int, max_processes: int = 4,
                                      visuals: bool = False) -> np.ndarray:
    """
    Basic method of 3-Dimensional averaging using parallel computations.
    Takes average value of all point around given point with given radius.
    Args:
        inputed_field (NDArray): field to get averaged
        radius (int): averaging radius around array point
        max_processes (int): maximum of processes to use
        visuals (bool): enables progress bar verbose
    Returns:
        NDArray: peasantly averaged 3d field
    """
    n, m, d = inputed_field.shape
    output_field = np.zeros((n, m, d))
    pool = multiprocessing.Pool(processes=max_processes)
    args_list = [(i, j, k, inputed_field, radius, average_this_3d_point)
                 for i in range(n) for j in range(m) for k in range(d)]
    chunksize = int(max([1, (n * m * d) / (4 * max_processes)]))
    if visuals:
        results = list(tqdm(pool.imap(process_func_3d, args_list, chunksize=chunksize),
                            total=(n * m * d), miniters=1000, ncols=100, leave=False,
                            position=0, desc="⚊ Iteration Progress"))
    else:
        results = list(pool.imap(process_func_3d,
                       args_list, chunksize=chunksize))
    pool.close()
    f = 0
    for i in range(0, n):
        for j in range(0, m):
            for k in range(0, d):
                output_field[i][j][k] = results[f]
                f += 1

    return output_field


@njit
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
    """
    Basic method of 2-Dimensional averaging. Takes average value of
    all point around given point with given radius.
    Args:
        inputed_field (NDArray): field to get averaged
        radius (int): averaging radius around array point
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
                    output_field[i][j] = average_this_2d_point(
                        i, j, inputed_field, radius)
                    pbar.update(1)
    else:
        for i in range(n):
            for j in range(m):
                output_field[i][j] = average_this_2d_point(
                    i, j, inputed_field, radius)
    return output_field


def process_func_2d(args):
    i, j, inputed_field, radius, average_this_2d_point_func = args
    return average_this_2d_point_func(i, j, inputed_field, radius)


def basic_2d_array_averaging_parallel(inputed_field: np.ndarray,
                                      radius: int, max_processes: int = 4,
                                      visuals: bool = False) -> np.ndarray:
    """
    Basic method of 2-Dimensional averaging using parallel computations.
    Takes average value of all point around given point with given radius.
    Args:
        inputed_field (NDArray): field to get averaged
        radius (int): averaging radius around array point
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
                            total=(n * m), miniters=1000, ncols=100, position=0,
                            leave=False, desc="⚊ Iteration Progress"))
    else:
        results = list(pool.imap(process_func_2d,
                       args_list, chunksize=chunksize))
    pool.close()
    k = 0
    for i in range(0, n):
        for j in range(0, m):
            output_field[i][j] = results[k]
            k += 1

    return output_field


def init_gauss_window(sigma: int) -> Tuple[np.ndarray, float]:
    """
    initing gauss window
    Args:
        sigma (int): sigma sets the radius of window and
        influence on blur coef in gauss formula
    Returns:
        Tuple[np.ndarray, float]: returning window and sum of elems in window
    """
    window_size = int(np.ceil(3 * sigma))
    window = np.zeros(2 * window_size + 1)

    s2 = 2 * sigma * sigma
    const = np.sqrt(2 * np.pi) * sigma

    window[window_size] = 1
    for i in range(1, window_size + 1):
        window[window_size - i] = window[window_size +
                                         i] = np.exp(- i * i / s2) / const
    window_sum = np.sum(window)

    return window, window_sum


def average_vertical_gauss_3d(copied_field: np.ndarray, window: np.ndarray, window_sum: float,
                              window_size: int, height: int, x: int, z: int):

    temp_vertical = np.zeros(height)
    for i in range(height):
        temp_vertical[i] = copied_field[z][i][x]

    for y in range(height):
        temp_elem = 0

        temp_index = y - window_size
        start = 0
        if temp_index < 0:
            start = window_size - y
            temp_index = 0

        for window_index in range(start, window_size * 2 + 1):

            if (temp_index < height):
                temp_elem += copied_field[z][temp_index][x] * \
                    window[window_index]
            temp_index += 1
            # sum += window[window_index]
            # if u ll put this ```sum +=window[window_index]``` in under if (in all gauss funcs)
            # then gauss will become limited by borders of initial array
        temp_vertical[y] = temp_elem / window_sum

    for i in range(height):
        copied_field[z][i][x] = temp_vertical[i]


def average_horizontal_gauss_3d(copied_field: np.ndarray, window: np.ndarray, window_sum: float,
                                window_size: int, width: int, y: int, z: int):

    temp_horizontal = np.zeros(width)
    for i in range(width):
        temp_horizontal[i] = copied_field[z][y][i]

    for x in range(width):
        temp_elem = 0

        temp_index = x - window_size
        start = 0
        if temp_index < 0:
            start = window_size - x
            temp_index = 0

        for window_index in range(start, window_size * 2 + 1):

            if (temp_index < width):
                temp_elem += copied_field[z][y][temp_index] * \
                    window[window_index]
            temp_index += 1

        temp_horizontal[x] = temp_elem / window_sum

    for i in range(width):
        copied_field[z][y][i] = temp_horizontal[i]


def average_depth_gauss_3d(copied_field: np.ndarray, window: np.ndarray, window_sum: float,
                           window_size: int, depth: int, x: int, y: int):

    temp_depth = np.zeros(depth)
    for i in range(depth):
        temp_depth[i] = copied_field[i][y][x]

    for z in range(depth):
        temp_elem = 0

        temp_index = z - window_size
        start = 0
        if temp_index < 0:
            start = window_size - z
            temp_index = 0

        for window_index in range(start, window_size * 2 + 1):

            if (temp_index < depth):
                temp_elem += copied_field[temp_index][y][x] * \
                    window[window_index]
            temp_index += 1

        temp_depth[z] = temp_elem / window_sum

    for i in range(depth):
        copied_field[i][y][x] = temp_depth[i]


def average_vertical_gauss_2d(copied_field: np.ndarray, window: np.ndarray, window_sum: float,
                              window_size: int, height: int, x: int):

    temp_vertical = np.zeros(height)
    for i in range(height):
        temp_vertical[i] = copied_field[i][x]

    for y in range(height):
        temp_elem = 0

        temp_index = y - window_size
        start = 0
        if temp_index < 0:
            start = window_size - y
            temp_index = 0

        for window_index in range(start, window_size * 2 + 1):

            if (temp_index < height):
                temp_elem += copied_field[temp_index][x] * window[window_index]
            temp_index += 1

        temp_vertical[y] = temp_elem / window_sum

    for i in range(height):
        copied_field[i][x] = temp_vertical[i]


def average_horizontal_gauss_2d(copied_field: np.ndarray, window: np.ndarray, window_sum: float,
                                window_size: int, width: int, y: int):

    temp_horizontal = np.zeros(width)
    for i in range(width):
        temp_horizontal[i] = copied_field[y][i]

    for x in range(width):
        temp_elem = 0

        temp_index = x - window_size
        start = 0
        if temp_index < 0:
            start = window_size - x
            temp_index = 0

        for window_index in range(start, window_size * 2 + 1):

            if (temp_index < width):
                temp_elem += copied_field[y][temp_index] * window[window_index]
            temp_index += 1

        temp_horizontal[x] = temp_elem / window_sum

    for i in range(width):
        copied_field[y][i] = temp_horizontal[i]


def average_3d_by_gauss(in_field: np.ndarray, sigma: int) -> np.ndarray:
    """
    Gauss method of 3-Dimensional averaging going line-by-line
    and doesnt take care of the border.
    Uses 'window' and 'sigma' to count the degree of averaging.
    Args:
        inputed_field (NDArray): field to get averaged
        sigma (int): defines the degree of averaging and size of window kernel,
                     which we impose on field by support functions like hor_aver, ver_aver and etc
    Returns:
        NDArray: new averaged 3d field
    """
    depth, height, width = in_field.shape

    copied_field = in_field.copy()

    window, window_sum = init_gauss_window(sigma)
    window_size = int(np.ceil(3 * sigma))

    for z in range(depth):

        for y in range(height):
            average_horizontal_gauss_3d(copied_field, window, window_sum,
                                        window_size, width, y, z)

        for x in range(width):
            average_vertical_gauss_3d(copied_field, window, window_sum,
                                      window_size, height, x, z)

    for x in range(width):
        for y in range(height):
            average_depth_gauss_3d(copied_field, window, window_sum,
                                   window_size, depth, x, y)

    return copied_field


def average_2d_by_gauss(in_field, sigma) -> np.ndarray:
    """
    Gauss method of 2-Dimensional averaging going line-by-line
    and doesnt take care of the border.
    Uses 'window' and 'sigma' to count the degree of averaging.
    Args:
        inputed_field (NDArray): field to get averaged
        sigma (int): defines the degree of averaging and size of window kernel,
                     which we impose on field
    Returns:
        NDArray: new averaged 2d field
    """
    height, width = in_field.shape

    copied_field = in_field.copy()

    window, window_sum = init_gauss_window(sigma)
    window_size = int(np.ceil(3 * sigma))

    for y in range(height):
        average_horizontal_gauss_2d(
            copied_field, window, window_sum, window_size, width, y)

    for x in range(width):
        average_vertical_gauss_2d(
            copied_field, window, window_sum, window_size, height, x)

    return copied_field


def basic_2d_averaging_iterations(in_field: np.ndarray, iterations_number: int = 1,
                                  radius: int = 1, processes: int = 1,
                                  iterations_visuals: bool = False,
                                  averaging_visuals: bool = False,
                                  leave: bool = True) -> np.ndarray:
    result = in_field
    if iterations_visuals:
        for i in tqdm(range(iterations_number), desc="⚊ Total Progress", position=1, leave=leave):
            result = basic_2d_array_averaging_parallel(np.asarray(result), radius=radius,
                                                       max_processes=processes,
                                                       visuals=averaging_visuals)
    else:
        for i in range(iterations_number):
            result = basic_2d_array_averaging_parallel(np.asarray(result), radius=radius,
                                                       max_processes=processes,
                                                       visuals=averaging_visuals)
    return (result)


def gauss_2d_averaging_iterations(in_field: np.ndarray, iterations_number: int = 1, radius: int = 1,
                                  processes: int = 1, iterations_visuals: bool = False,
                                  averaging_visuals: bool = False) -> np.ndarray:
    result = in_field
    if iterations_visuals:
        for i in tqdm(range(iterations_number)):
            result = average_2d_by_gauss(np.asarray(result), sigma=radius)
    else:
        for i in range(iterations_number):
            result = average_2d_by_gauss(np.asarray(result), sigma=radius)
    return (result)


def basic_3d_averaging_iterations(in_field: np.ndarray, iterations_number: int = 1,
                                  radius: int = 1, processes: int = 1,
                                  iterations_visuals: bool = False,
                                  averaging_visuals: bool = False,
                                  leave: bool = True) -> np.ndarray:
    result = in_field
    if iterations_visuals:
        for i in tqdm(range(iterations_number), desc="⚊ Total Progress", position=1, leave=leave):
            result = basic_3d_array_averaging_parallel(np.asarray(result), radius=radius,
                                                       max_processes=processes,
                                                       visuals=averaging_visuals)
    else:
        for i in range(iterations_number):
            result = basic_3d_array_averaging_parallel(np.asarray(result), radius=radius,
                                                       max_processes=processes,
                                                       visuals=averaging_visuals)
    return (result)


def gauss_3d_averaging_iterations(in_field: np.ndarray, iterations_number: int = 1, radius: int = 1,
                                  processes: int = 1, iterations_visuals: bool = False,
                                  averaging_visuals: bool = False) -> np.ndarray:
    result = in_field
    if iterations_visuals:
        for i in tqdm(range(iterations_number)):
            result = average_3d_by_gauss(np.asarray(result), sigma=radius)
    else:
        for i in range(iterations_number):
            result = average_3d_by_gauss(np.asarray(result), sigma=radius)
    return (result)


def test():
    averaging_width = 1
    w, h = 5, 3
    input_field = [[float(0) for y in range(h)] for x in range(w)]
    input_field[4][2] = 15
    print(input_field)
    print(basic_2d_array_averaging(input_field, averaging_width))
