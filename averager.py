import multiprocessing
import numpy as np
from tqdm import tqdm


def average_this_3d_point(i: int, j: int, k: int, in_field: np.ndarray, radius: int) -> float:
    """
    Basic method of 3-Dimensional averaging. Takes average value of
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
    n, m, d = in_field.shape
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
    """
    Function takes field and use basic 3d averaging method. Gives back averaged field

    Args:
        inputed_field (np.ndarray): field to get averaged
        radius (int): averaging radius around this point

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
                    output_field[i][j][k] = average_this_3d_point(i, j, k, inputed_field, radius)
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
        radius (int): averaging radius around this point
        max_processes (int): maximum of processes to use
        visuals (bool): enables progress bar verbose

    Returns:
        NDArray: peasantly averaged 2d field
    """
    n, m, d = inputed_field.shape
    output_field = np.zeros((n, m, d))
    pool = multiprocessing.Pool(processes=max_processes)
    args_list = [(i, j, k, inputed_field, radius, average_this_3d_point)
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
                output_field[i][j][k] = results[f]
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
    """
    Basic method of 2-Dimensional averaging. Takes average value of
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
    """
    Basic method of 2-Dimensional averaging using parallel computations.
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


def gauss_wnd_init(sigma: int) -> np.ndarray:

    wnd_sz = np.ceil(3 * sigma)
    window = np.zeros(2 * wnd_sz + 1)
    s2 = 2 * sigma * sigma
    const = np.sqrt(2 * np.pi) * sigma

    window[wnd_sz] = 1
    for i in range(1, wnd_sz + 1):
        window[wnd_sz - i] = window[wnd_sz + i] = np.exp(- i * i / s2) / const

    return window


def ver_aver_3(mtx: np.ndarray, window: np.ndarray, wnd_sz: int, MAX_VER: int, x: int, z: int):

    t_mtx = np.zeros(MAX_VER)
    for i in range(MAX_VER):
        t_mtx[i] = mtx[z][i][x]

    for y in range(MAX_VER):
        sum = 0
        t_elem = 0

        for w_ind in range(0, wnd_sz * 2 + 1):
            t_ind = w_ind - wnd_sz + y

            if (t_ind >= 0 and t_ind < MAX_VER):
                t_elem += mtx[z][t_ind][x] * window[w_ind]
            sum += window[w_ind]
            # if u ll put this ```sum +=window[w_ind]``` in under if (in all gauss funcs)
            # then gauss will become limited by borders of initial array
        if (sum != 0):
            t_mtx[y] = t_elem / sum

    for i in range(MAX_VER):
        mtx[z][i][x] = t_mtx[i]


def hor_aver_3(mtx: np.ndarray, window: np.ndarray, wnd_sz: int, MAX_HOR: int, y: int, z: int):

    t_mtx = np.zeros(MAX_HOR)
    for i in range(MAX_HOR):
        t_mtx[i] = mtx[z][y][i]

    for x in range(MAX_HOR):
        sum = 0
        t_elem = 0

        for w_ind in range(0, wnd_sz * 2 + 1):
            t_ind = w_ind - wnd_sz + x

            if (t_ind >= 0 and t_ind < MAX_HOR):
                t_elem += mtx[z][y][t_ind] * window[w_ind]
            sum += window[w_ind]

        if (sum != 0):
            t_mtx[x] = t_elem / sum

    for i in range(MAX_HOR):
        mtx[z][y][i] = t_mtx[i]


def dep_aver_3(mtx: np.ndarray, window: np.ndarray, wnd_sz: int, MAX_DEP: int, x: int, y: int):

    t_mtx = np.zeros(MAX_DEP)
    for i in range(MAX_DEP):
        t_mtx[i] = mtx[i][y][x]

    for z in range(MAX_DEP):
        sum = 0
        t_elem = 0

        for w_ind in range(0, wnd_sz * 2 + 1):
            t_ind = w_ind - wnd_sz + z

            if (t_ind >= 0 and t_ind < MAX_DEP):
                t_elem += mtx[t_ind][y][x] * window[w_ind]
            sum += window[w_ind]

        if (sum != 0):
            t_mtx[z] = t_elem / sum

    for i in range(MAX_DEP):
        mtx[i][y][x] = t_mtx[i]


def ver_aver_2(mtx: np.ndarray, window: np.ndarray, wnd_sz: int, MAX_VER: int, x: int):

    t_mtx = np.zeros(MAX_VER)
    for i in range(MAX_VER):
        t_mtx[i] = mtx[i][x]

    for y in range(MAX_VER):
        sum = 0
        t_elem = 0

        for w_ind in range(0, wnd_sz * 2 + 1):
            t_ind = w_ind - wnd_sz + y

            if (t_ind >= 0 and t_ind < MAX_VER):
                t_elem += mtx[t_ind][x] * window[w_ind]
            sum += window[w_ind]

        if (sum != 0):
            t_mtx[y] = t_elem / sum

    for i in range(MAX_VER):
        mtx[i][x] = t_mtx[i]


def hor_aver_2(mtx: np.ndarray, window: np.ndarray, wnd_sz: int, MAX_HOR: int, y: int):

    t_mtx = np.zeros(MAX_HOR)
    for i in range(MAX_HOR):
        t_mtx[i] = mtx[y][i]

    for x in range(MAX_HOR):
        sum = 0
        t_elem = 0

        for w_ind in range(0, wnd_sz * 2 + 1):
            t_ind = w_ind - wnd_sz + x

            if (t_ind >= 0 and t_ind < MAX_HOR):
                t_elem += mtx[y][t_ind] * window[w_ind]
            sum += window[w_ind]

        if (sum != 0):
            t_mtx[x] = t_elem / sum

    for i in range(MAX_HOR):
        mtx[y][i] = t_mtx[i]


def gauss_3d(in_field: np.ndarray, sigma: int) -> np.ndarray:
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
    MAX_DEP, MAX_VER, MAX_HOR = in_field.shape

    mtx = in_field.copy()

    window = gauss_wnd_init(sigma)
    wnd_sz = np.ceil(3 * sigma)

    for z in range(MAX_DEP):

        for y in range(MAX_VER):
            hor_aver_3(mtx, window, wnd_sz, MAX_HOR, y, z)

        for x in range(MAX_HOR):
            ver_aver_3(mtx, window, wnd_sz, MAX_VER, x, z)

    for x in range(MAX_HOR):
        for y in range(MAX_VER):
            dep_aver_3(mtx, window, wnd_sz, MAX_DEP, x, y)

    return mtx


def gauss_2d(in_field, sigma) -> np.ndarray:
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
    MAX_VER, MAX_HOR = in_field.shape

    mtx = in_field.copy()

    window = gauss_wnd_init(sigma)
    wnd_sz = np.ceil(3 * sigma)

    for y in range(MAX_VER):
        hor_aver_2(mtx, window, wnd_sz, MAX_HOR, y)

    for x in range(MAX_HOR):
        ver_aver_2(mtx, window, wnd_sz, MAX_VER, x)

    return mtx


def test():
    averaging_width = 1
    w, h = 5, 3
    input_field = [[float(0) for y in range(h)] for x in range(w)]
    input_field[4][2] = 15
    print(input_field)
    print(basic_2d_array_averaging(input_field, averaging_width))
