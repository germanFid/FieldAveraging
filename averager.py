import math


def gauss_wnd_init(sigma):

    wnd_sz = math.ceil(3 * sigma)
    window = [0] * (2 * wnd_sz + 1)
    s2 = 2 * sigma * sigma
    const = math.sqrt(2 * math.pi) * sigma

    window[wnd_sz] = 1
    for i in range(1, wnd_sz + 1):
        window[wnd_sz - i] = window[wnd_sz + i] = math.exp(- i * i / s2) / const

    return window


def ver_aver_3(mtx, window, wnd_sz, MAX_VER, x, z):

    for y in range(MAX_VER):
        sum = 0
        t_elem = 0

        for w_ind in range(0, wnd_sz * 2 + 1):
            t_ind = w_ind - wnd_sz + y

            if (t_ind >= 0 and t_ind < MAX_VER):
                t_elem += mtx[z][t_ind][x] * window[w_ind]
            sum += window[w_ind]
            # if u ll put this ```sum +=window[w_ind]``` in closest if above (in all gauss funcs)
            # then gauss will become limited by borders of initial array
        if (sum != 0):
            mtx[z][y][x] = t_elem / sum


def hor_aver_3(mtx, window, wnd_sz, MAX_HOR, y, z):

    for x in range(MAX_HOR):
        sum = 0
        t_elem = 0

        for w_ind in range(0, wnd_sz * 2 + 1):
            t_ind = w_ind - wnd_sz + x

            if (t_ind >= 0 and t_ind < MAX_HOR):
                t_elem += mtx[z][y][t_ind] * window[w_ind]
            sum += window[w_ind]

        if (sum != 0):
            mtx[z][y][x] = t_elem / sum


def dep_aver_3(mtx, window, wnd_sz, MAX_DEP, x, y):

    for z in range(MAX_DEP):
        sum = 0
        t_elem = 0

        for w_ind in range(0, wnd_sz * 2 + 1):
            t_ind = w_ind - wnd_sz + z

            if (t_ind >= 0 and t_ind < MAX_DEP):
                t_elem += mtx[t_ind][y][x] * window[w_ind]
            sum += window[w_ind]

        if (sum != 0):
            mtx[z][y][x] = t_elem / sum


def ver_aver_2(mtx, window, wnd_sz, MAX_VER, x):

    for y in range(MAX_VER):
        sum = 0
        t_elem = 0

        for w_ind in range(0, wnd_sz * 2 + 1):
            t_ind = w_ind - wnd_sz + y

            if (t_ind >= 0 and t_ind < MAX_VER):
                t_elem += mtx[t_ind][x] * window[w_ind]
            sum += window[w_ind]

        if (sum != 0):
            mtx[y][x] = t_elem / sum


def hor_aver_2(mtx, window, wnd_sz, MAX_HOR, y):

    for x in range(MAX_HOR):
        sum = 0
        t_elem = 0

        for w_ind in range(0, wnd_sz * 2 + 1):
            t_ind = w_ind - wnd_sz + x

            if (t_ind >= 0 and t_ind < MAX_HOR):
                t_elem += mtx[y][t_ind] * window[w_ind]
            sum += window[w_ind]

        if (sum != 0):
            mtx[y][x] = t_elem / sum


def gauss_3d(in_field, sigma):

    MAX_DEP = len(in_field)
    MAX_VER = len(in_field[0])
    MAX_HOR = len(in_field[0][0])

    mtx = in_field.copy()

    window = gauss_wnd_init(sigma)
    wnd_sz = math.ceil(3 * sigma)

    for z in range(MAX_DEP):

        for y in range(MAX_VER):
            hor_aver_3(mtx, window, wnd_sz, MAX_HOR, y, z)

        for x in range(MAX_HOR):
            ver_aver_3(mtx, window, wnd_sz, MAX_VER, x, z)

    for x in range(MAX_HOR):
        for y in range(MAX_VER):
            dep_aver_3(mtx, window, wnd_sz, MAX_DEP, x, y)

    return mtx


def gauss_2d(in_field, sigma):

    MAX_VER = len(in_field)
    MAX_HOR = len(in_field[0])

    mtx = in_field.copy()

    window = gauss_wnd_init(sigma)
    wnd_sz = math.ceil(3 * sigma)

    for y in range(MAX_VER):
        hor_aver_2(mtx, window, wnd_sz, MAX_HOR, y)

    for x in range(MAX_HOR):
        ver_aver_2(mtx, window, wnd_sz, MAX_VER, x)

    return mtx


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


def average_this_2d_point(i: int, j: int, in_field: list, radius: int) -> float:
    """Basic method of 2-Dimensional averaging. Takes average value of
    all point around given point with given radius.

    Args:
        i (int): index in row
        j (int): index in column
        in_field (list): field to get average value from
        radius (int): averaging radius around this point

    Returns:
        float: peasantly averaged value of our 2d point in field
    """

    n = len(in_field)
    m = len(in_field[0])
    i_set = get_start_end_lenght(i, radius, n)
    j_set = get_start_end_lenght(j, radius, m)
    sum_of_elements = 0.0
    number_of_elements = i_set[2] * j_set[2]
    for ii in range(i_set[0], i_set[1] + 1):
        for jj in range(j_set[0], j_set[1] + 1):
            sum_of_elements = sum_of_elements + in_field[ii][jj]
    average_value = sum_of_elements / number_of_elements

    return average_value


def basic_2d_array_averaging(inputed_field: list, radius: int) -> list:
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
    for i in range(0, n):
        for j in range(0, m):
            output_field[i][j] = average_this_2d_point(i, j, inputed_field, radius)
    return output_field


def test():
    averaging_width = 1
    w, h = 5, 3
    input_field = [[float(0) for y in range(h)] for x in range(w)]
    input_field[4][2] = 15
    print(input_field)
    print(basic_2d_array_averaging(input_field, averaging_width))
