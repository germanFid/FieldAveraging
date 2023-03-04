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


def test():
    averaging_width = 1
    w, h = 5, 3
    input_field = [[float(0) for x in range(w)] for y in range(h)]
    input_field[2][4] = 15
    print(input_field)
    #print(line_2_row_averaging_this(input_field, averaging_width))
    print(diagonally_averaging_this(input_field, averaging_width))

