import math


def gauss_wnd_init(size, sigma):

    window = [0] * (2 * size + 1)
    s2 = 2 * sigma * sigma
    const = math.sqrt(2 * math.pi) * sigma

    window[size] = 1
    for i in range(1, size + 1):
        window[size - i] = window[size + i] = math.exp(- i * i / s2) / const

    return window


def ver_aver_3(mtx, t_mtx, window, wnd_mid, wnd_sz, MAX_VER, x, z):

    for y in range(MAX_VER):
        sum = 0
        t_elem = 0

        for k in range(-wnd_sz, wnd_sz + 1):
            t_ind = k + y
            w_ind = k + wnd_mid

            if (t_ind >= 0 and t_ind < MAX_VER):
                t_elem += mtx[z][t_ind][x] * window[w_ind]
                sum += window[w_ind]

        if (sum != 0):
            t_mtx[z][y][x] = t_elem / sum


def hor_aver_3(mtx, t_mtx, window, wnd_mid, wnd_sz, MAX_HOR, y, z):

    for x in range(MAX_HOR):
        sum = 0
        t_elem = 0

        for k in range(-wnd_sz, wnd_sz + 1):
            t_ind = k + x
            w_ind = k + wnd_mid

            if (t_ind >= 0 and t_ind < MAX_HOR):
                t_elem += mtx[z][y][t_ind] * window[w_ind]
                sum += window[w_ind]

        if (sum != 0):
            t_mtx[z][y][x] = t_elem / sum


def dep_aver_3(mtx, t_mtx, window, wnd_mid, wnd_sz, MAX_DEP, x, y):

    for z in range(MAX_DEP):
        sum = 0
        t_elem = 0

        for k in range(-wnd_sz, wnd_sz + 1):
            t_ind = k + z
            w_ind = k + wnd_mid

            if (t_ind >= 0 and t_ind < MAX_DEP):
                t_elem += mtx[t_ind][y][x] * window[w_ind]
                sum += window[w_ind]

        if (sum != 0):
            t_mtx[z][y][x] = t_elem / sum


def ver_aver_2(mtx, t_mtx, window, wnd_mid, wnd_sz, MAX_VER, x):

    for y in range(MAX_VER):
        sum = 0
        t_elem = 0

        for k in range(-wnd_sz, wnd_sz + 1):
            t_ind = k + y
            w_ind = k + wnd_mid

            if (t_ind >= 0 and t_ind < MAX_VER):
                t_elem += mtx[t_ind][x] * window[w_ind]
                sum += window[w_ind]

        if (sum != 0):
            t_mtx[y][x] = t_elem / sum


def hor_aver_2(mtx, t_mtx, window, wnd_mid, wnd_sz, MAX_HOR, y):

    for x in range(MAX_HOR):
        sum = 0
        t_elem = 0

        for k in range(-wnd_sz, wnd_sz + 1):
            t_ind = k + x
            w_ind = k + wnd_mid

            if (t_ind >= 0 and t_ind < MAX_HOR):
                t_elem += mtx[y][t_ind] * window[w_ind]
                sum += window[w_ind]

        if (sum != 0):
            t_mtx[y][x] = t_elem / sum


def gauss_3d(in_field, sigma):

    MAX_DEP = len(in_field)
    MAX_VER = len(in_field[0])
    MAX_HOR = len(in_field[0][0])

    mtx = in_field.copy()
    t_mtx = in_field.copy()

    wnd_mid = 25000
    window = gauss_wnd_init(wnd_mid, sigma)

    wnd_sz = math.ceil(3 * sigma)

    if (wnd_sz > wnd_mid):
        print("Too big sigma")
        exit()

    for z in range(MAX_DEP):

        for y in range(MAX_VER):
            hor_aver_3(mtx, t_mtx, window, wnd_mid, wnd_sz, MAX_HOR, y, z)

        mtx = t_mtx.copy()

        for x in range(MAX_HOR):
            ver_aver_3(mtx, t_mtx, window, wnd_mid, wnd_sz, MAX_VER, x, z)

        mtx = t_mtx.copy()

    for x in range(MAX_HOR):
        for y in range(MAX_VER):
            dep_aver_3(mtx, t_mtx, window, wnd_mid, wnd_sz, MAX_DEP, x, y)

    mtx = t_mtx.copy()

    return mtx


def gauss_2d(in_field, sigma):

    MAX_VER = len(in_field)
    MAX_HOR = len(in_field[0])

    mtx = in_field.copy()
    t_mtx = in_field.copy()

    wnd_mid = 25000
    window = gauss_wnd_init(wnd_mid, sigma)

    wnd_sz = math.ceil(3 * sigma)

    if (wnd_sz > wnd_mid):
        print("Too big sigma, the biggest possible was taken")
        wnd_sz = wnd_mid

    for y in range(MAX_VER):
        hor_aver_2(mtx, t_mtx, window, wnd_mid, wnd_sz, MAX_HOR, y)

    mtx = t_mtx.copy()

    for x in range(MAX_HOR):
        ver_aver_2(mtx, t_mtx, window, wnd_mid, wnd_sz, MAX_VER, x)

    mtx = t_mtx.copy()

    return mtx
