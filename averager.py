def Average_This_Point(i, j, In_Field, k):

    m = len(In_Field[0])
    n = len(In_Field)

    i_start = i-k
    if i_start < 0:
        i_start = 0
    j_start = j-k
    if j_start < 0:
        j_start = 0
    i_end = i+k
    if i_end > n:
        i_end = n
    j_end = j+k
    if j_end > m:
        j_end = m

    # print("\t>>", i, j, i_start, i_end, j_start,  j_end, n, m)
    Number_Of_Elements = (i_end-i_start + 1) * (j_end-j_start + 1)
    Sum_Of_Elements = 0
    for ii in range(i_start, i_end):
        for jj in range(j_start, j_end):
            Sum_Of_Elements = Sum_Of_Elements+In_Field[ii][jj]

    Average_Value = Sum_Of_Elements / Number_Of_Elements

    ij_value = Average_Value

    return ij_value


def Line_2_Row_Averaging_This(Inputed_Field, k):
    
    m = len(Inputed_Field[0])
    n = len(Inputed_Field)

    Output_Field = [[0 for x in range(m)] for y in range(n)]

    for i in range(0, n):
        for j in range(0, m):

            Output_Field[i][j] = Average_This_Point(i, j, Inputed_Field, k)

    return Output_Field


def Diagonally_Averaging_This(Inputed_Field, k):

    n = len(Inputed_Field[0])
    m = len(Inputed_Field)

    Output_Field = [[0 for x in range(n)] for y in range(m)]

    for i_begin in range(0, n):
        i = i_begin
        j = 0
        while i >= 0 and j <= m-1:

            Output_Field[j][i] = Average_This_Point(i, j, Inputed_Field, k)

            if (i >= 0):
                i = i-1
            if (j <= m-1):
                j = j+1

    for j_begin in range(1, m):
        j = j_begin
        i = n-1
        while i >= 0 and j <= m-1:

            Output_Field[j][i] = Average_This_Point(i, j, Inputed_Field, k)

            if (i >= 0):
                i = i-1
            if (j <= m-1):
                j = j+1

    return Output_Field


def Test():
    Averaging_Width = 1
    w, h = 5, 3
    Input_Field = [[0 for x in range(w)] for y in range(h)]
    Input_Field[0][0] = 15
    print(Input_Field)
    Temp_Field=Input_Field
    print(Line_2_Row_Averaging_This(Temp_Field, Averaging_Width))

    Temp_Field=Input_Field
    print(Diagonally_Averaging_This(Temp_Field, Averaging_Width))

