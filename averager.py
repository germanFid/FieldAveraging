import math

def average_this_point(j, i, in_field, k):
    
    n = len(in_field[0])
    m = len(in_field)
    
    i_start = i-k
    if i_start < 0:
        i_start = 0
    j_start = j-k
    if j_start < 0:
        j_start = 0
    i_end = i+k
    if i_end > n-1:
        i_end = n-1
    j_end = j+k
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
            sum_of_elements = sum_of_elements+in_field[jj][ii]
    average_value = sum_of_elements / number_of_elements
    ij_value = average_value
    return ij_value


def line_2_row_averaging_this(inputed_field, k):
    
    n = len(inputed_field[0])
    m = len(inputed_field)

    output_field = [[float(0) for x in range(n)] for y in range(m)]
    print(output_field) 
    for i in range(0, n):
        for j in range(0, m):

            output_field[j][i] = average_this_point(j, i, inputed_field, k)

    return output_field


def diagonally_averaging_this(inputed_field, k):

    n = len(inputed_field[0])
    m = len(inputed_field)

    output_field = [[float(0) for x in range(n)] for y in range(m)]
    
    for i_begin in range(0, n):
        i = i_begin
        j = 0
        while i >= 0 and j <= m-1:
            output_field[j][i] = average_this_point(j, i, inputed_field, k)

            if i >= 0:
                i = i-1
            if j <= m-1:
                j = j+1

    for j_begin in range(1, m):
        j = j_begin
        i = n-1
        while i >= 0 and j <= m-1:
            output_field[j][i] = average_this_point(j, i, inputed_field, k)

            if i >= 0:
                i = i-1
            if j <= m-1:
                j = j+1

    return output_field


def gauss_method(in_field, radius):
    

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


    # # testing output
    # sum = 0
    # for z in range(MAX_DEP):
    #     for y in range(MAX_VER):
    #         for x in range(MAX_HOR):
    #             sum += in_field[z][y][x]    
    #             # print('%.2f' % in_field[y][x], end = "  ")
    #         # print()
    # print(sum)



def test():
    averaging_width = 1
    w, h = 5, 3
    input_field = [[float(0) for x in range(w)] for y in range(h)]
    input_field[2][4] = 15
    print(input_field)
    #print(line_2_row_averaging_this(input_field, averaging_width))
    print(diagonally_averaging_this(input_field, averaging_width))

