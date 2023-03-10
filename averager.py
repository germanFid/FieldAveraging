def gauss_method(in_field, radius):
    import math

    MAX_DEP = len(in_field)
    MAX_VER = len(in_field[0])
    MAX_HOR = len(in_field[0][0])

#-------------------------
    sigma = 5
    s2 = 2 * sigma * sigma
#-------------------------
   
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
                        
                    if(t_ind >= 0 and t_ind < MAX_HOR):
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

                    if(t_ind >= 0 and t_ind < MAX_VER):
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

                    if(t_ind >= 0 and t_ind < MAX_VER):
                        t_elem += output_field[t_ind][y][x] * window[k + wnd_mid]
                        sum += window[k + wnd_mid]
                
                tmp_dep[z] = t_elem / sum
            
            for t in range(MAX_DEP):
                t_mtx[t][y][x] = tmp_dep[t]

        output_field = t_mtx.copy()
    
    return output_field


def average_this_3d_point(i: int, j: int, k: int, in_field: list, radius: int) -> float:
    """Basic method of 3-Dimensional averaging. Takes average value of all point around given point with given radius.

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

    i_start = i-radius
    if i_start < 0:
        i_start = 0
    j_start = j-radius
    if j_start < 0:
        j_start = 0
    k_start = k-radius
    if k_start < 0:
        k_start = 0
    i_end = i+radius
    if i_end > n-1:
        i_end = n-1
    j_end = j+radius
    if j_end > m-1:
        j_end = m-1
    k_end = k+radius
    if k_end > d-1:
        k_end = d-1

    height = j_end-j_start+1
    width = i_end-i_start+1
    depth = k_end-k_start+1
    if height == 0:
        height = 1
    if width == 0:
        width = 1
    if depth == 0:
        depth = 1

    number_of_elements = width * height * depth
    sum_of_elements = 0.0
    for ii in range(i_start, i_end+1):
        for jj in range(j_start, j_end+1):
            for kk in range(k_start, k_end+1):    
                sum_of_elements = sum_of_elements+in_field[ii][jj][kk]
    average_value = sum_of_elements / number_of_elements
    ijk_value = average_value

    return ijk_value


def line_2_row_averaging_this_3d(inputed_field: list, radius: int) -> list:
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
    """Basic method of 2-Dimensional averaging. Takes average value of all point around given point with given radius.

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
    
    i_start = i-radius
    if i_start < 0:
        i_start = 0
    j_start = j-radius
    if j_start < 0:
        j_start = 0
    i_end = i+radius
    if i_end > n-1:
        i_end = n-1
    j_end = j+radius
    if j_end > m-1:
        j_end = m-1
    height = j_end-j_start+1
    width = i_end-i_start+1
    if height == 0:
        height = 1
    if width == 0:
        width = 1
    number_of_elements = width * height
    sum_of_elements = 0.0
    for ii in range(i_start, i_end+1):
        for jj in range(j_start, j_end+1):
            print(len(in_field), len(in_field[0]), ii, jj)
            sum_of_elements = sum_of_elements+in_field[ii][jj]
    average_value = sum_of_elements / number_of_elements
    ij_value = average_value

    return ij_value


def line_2_row_averaging_this_2d(inputed_field: list, radius: int) -> list:
    """Basic method of 2-Dimensional averaging. Takes average value of all point around given point with given radius.

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
    print(line_2_row_averaging_this_2d(input_field, averaging_width))
    print(gauss_method(input_field, averaging_width))

