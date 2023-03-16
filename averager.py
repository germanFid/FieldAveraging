import multiprocessing
import numpy as np
from tqdm import tqdm

def gauss_method(in_field, radius):
    import math

    MAX_DEP = len(in_field)
    MAX_VER = len(in_field[0])
    MAX_HOR = len(in_field[0][0])

# -------------------------
    sigma = 5
    s2 = 2 * sigma * sigma
# -------------------------

    # matrix init
    t_mtx = [0] * MAX_DEP
    output_field = [0] * MAX_DEP

    for i in range(MAX_DEP):
        t_mtx[i] = [0] * MAX_DEP
        output_field[i] = [0] * MAX_DEP

    for j in range(MAX_DEP):
        for i in range(MAX_VER):
            t_mtx[j][i] = [0] * MAX_HOR
            output_field[j][i] = [0] * MAX_HOR

    output_field = in_field.copy()

    wnd_mid = wnd_max_sz = max(max(MAX_DEP, MAX_HOR), MAX_VER)
    window = [0] * (2 * wnd_max_sz + 1)
    window[wnd_mid] = 1
    wnd_sz = math.ceil(3 * radius)

    # temp arrays init of directions size
    tmp_hor = [0] * MAX_HOR
    tmp_ver = [0] * MAX_VER
    tmp_dep = [0] * MAX_DEP

    # window init
    for i in range(1, wnd_sz + 1):
        window[wnd_mid - i] = window[wnd_mid + i] = math.exp(- i * i / s2)

    for z in range(MAX_DEP):
        # hor averagin first
        for y in range(MAX_VER):
            for x in range(MAX_HOR):
                '''
                    we ll count summ of used normalized coefs.
                    we have to do it each time cuz for averagin
                    border elems we use only part of matrix
                '''
                sum = 0
                t_elem = 0
                # going with window around this elem
                for k in range(-wnd_sz, wnd_sz + 1):
                    # temp index of nearest ones
                    t_ind = x + k

                    if (t_ind >= 0 and t_ind < MAX_HOR):
                        t_elem += output_field[z][y][t_ind] * window[k + wnd_mid]
                        sum += window[k + wnd_mid]

                tmp_hor[x] = t_elem / sum

            for t in range(MAX_HOR):
                t_mtx[z][y][t] = tmp_hor[t]

        # important to copy after we've changed output_field
        output_field = t_mtx.copy()

        # ver aver sec
        for x in range(MAX_HOR):
            for y in range(MAX_VER):
                sum = 0
                t_elem = 0

                for k in range(-wnd_sz, wnd_sz + 1):
                    t_ind = y + k

                    if (t_ind >= 0 and t_ind < MAX_VER):
                        t_elem += output_field[z][t_ind][x] * window[k + wnd_mid]
                        sum += window[k + wnd_mid]

                tmp_ver[y] = t_elem / sum

            for t in range(MAX_VER):
                t_mtx[z][t][x] = tmp_ver[t]

        output_field = t_mtx.copy()

        # depth aver
        for x in range(MAX_HOR):
            for y in range(MAX_VER):
                sum = 0
                t_elem = 0

                for k in range(-wnd_sz, wnd_sz + 1):
                    t_ind = z + k

                    if (t_ind >= 0 and t_ind < MAX_VER):
                        t_elem += output_field[t_ind][y][x] * window[k + wnd_mid]
                        sum += window[k + wnd_mid]

                tmp_dep[z] = t_elem / sum

            for t in range(MAX_DEP):
                t_mtx[t][y][x] = tmp_dep[t]

        output_field = t_mtx.copy()

    return output_field


def get_start_end_lenght(coordinate: int, radius: int, max_coordinate: int) -> list:
    start_end_lenght = list()
    start_end_lenght.append(coordinate - radius)
    start_end_lenght.append(coordinate + radius)
    if start_end_lenght[0] < 0:
        start_end_lenght[0] = 0
    if start_end_lenght[1] > max_coordinate - 1:
        start_end_lenght[1] = max_coordinate - 1
    start_end_lenght.append(start_end_lenght[1] - start_end_lenght[0] + 1)
    if start_end_lenght[2] == 0:
        start_end_lenght[2] = 1
    return start_end_lenght


def average_this_3d_point(i: int, j: int, k: int, in_field: list, radius: int) -> float:
    """Basic method of 3-Dimensional averaging. Takes average value of
    all point around given point with given radius.

    Args:
        i (int): index in row
        j (int): index in column
        k (int): index in depth
        in_field (list): field to get average value from
        radius (int): averaging radius around this point

    Returns:
        float: peasantly averaged value of our 3d point in field
    """
    n = len(in_field)
    m = len(in_field[0])
    d = len(in_field[0][0])
    i_set = get_start_end_lenght(i, radius, n)
    j_set = get_start_end_lenght(j, radius, m)
    k_set = get_start_end_lenght(k, radius, d)
    sum_of_elements = 0.0
    number_of_elements = i_set[2] * j_set[2] * k_set[2]
    for ii in range(i_set[0], i_set[1] + 1):
        for jj in range(j_set[0], j_set[1] + 1):
            for kk in range(k_set[0], k_set[1] + 1):
                sum_of_elements = sum_of_elements + in_field[ii][jj][kk]
    average_value = sum_of_elements / number_of_elements
    ijk_value = average_value

    return ijk_value


def basic_3d_array_averaging(inputed_field: list, radius: int) -> list:
    """Function takes field and use basic 3d averaging method. Gives back averaged field

    Args:
        inputed_field (list): field to get averaged
        radius (int): averaging radius around this point

    Returns:
        list: peasantly averaged 3d field
    """
    n = len(inputed_field)
    m = len(inputed_field[0])
    d = len(inputed_field[0][0])

    output_field = [[[float(0) for z in range(d)] for y in range(m)] for x in range(n)]
    for i in range(0, n):
        for j in range(0, m):
            for k in range(0, d):
                output_field[i][j][k] = average_this_3d_point(i, j, k, inputed_field, radius)
    return output_field


# def average_this_2d_point(i: int, j: int, in_field: list, radius: int) -> float:
#     """Basic method of 2-Dimensional averaging. Takes average value of
#     all point around given point with given radius.

#     Args:
#         i (int): index in row
#         j (int): index in column
#         in_field (list): field to get average value from
#         radius (int): averaging radius around this point

#     Returns:
#         float: peasantly averaged value of our 2d point in field
#     """

#     n = len(in_field)
#     m = len(in_field[0])
#     i_set = get_start_end_lenght(i, radius, n)
#     j_set = get_start_end_lenght(j, radius, m)
#     sum_of_elements = 0.0
#     number_of_elements = i_set[2] * j_set[2]
#     for ii in range(i_set[0], i_set[1] + 1):
#         for jj in range(j_set[0], j_set[1] + 1):
#             sum_of_elements = sum_of_elements + in_field[ii][jj]
#     average_value = sum_of_elements / number_of_elements

#     return average_value


def average_this_2d_point(i: int, j: int, in_field: np.ndarray, radius: int) -> float:
    n, m = in_field.shape

    i_start = max(0, i - radius)
    i_end = min(n - 1, i + radius)

    j_start = max(0, j - radius)
    j_end = min(m - 1, j + radius)

    window_size = (i_end - i_start + 1) * (j_end - j_start + 1)
    window_sum = np.sum(in_field[i_start:i_end + 1, j_start:j_end + 1])

    return window_sum / window_size


def basic_2d_array_averaging(inputed_field: np.ndarray, radius: int, visuals: bool = False) -> list:
    """Basic method of 2-Dimensional averaging. Takes average value of
    all point around given point with given radius.

    Args:
        inputed_field (list): field to get averaged
        radius (int): averaging radius around this point

    Returns:
        list: peasantly averaged 2d field
    """
    n = len(inputed_field)
    m = len(inputed_field[0])

    output_field = [[float(0) for y in range(m)] for x in range(n)]

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


def process_func(args):
    i, j, inputed_field, radius, average_this_2d_point_func = args
    return average_this_2d_point_func(i, j, inputed_field, radius)


def basic_2d_array_averaging_parallel(inputed_field: np.ndarray,
                                      radius: int, max_processes: int,
                                      visuals: bool = False) -> list:
    n = len(inputed_field)
    m = len(inputed_field[0])

    output_field = [[float(0) for y in range(m)] for x in range(n)]

    pool = multiprocessing.Pool(processes=max_processes)
    args_list = [(i, j, inputed_field, radius, average_this_2d_point)
                 for i in range(n) for j in range(m)]

    chunksize = int(max([1, (n * m) / (4 * max_processes)]))

    if visuals:
        results = list(tqdm(pool.imap_unordered(process_func, args_list, chunksize=chunksize),
                            total=(n * m), miniters=1000))

    else:
        results = list(pool.imap_unordered(process_func, args_list, chunksize=chunksize),
                       total=(n * m), miniters=1000)
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
