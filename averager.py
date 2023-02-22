def AverageThisPoint(i, j, In_Field, k):

    m = len(In_Field[0])
    n = len(In_Field)

    i_start = i-k
    if (i_start < 0):
        i_start = 0
    j_start = j-k
    if (j_start < 0):
        j_start = 0
    i_end = i+k
    if (i_end > n):
        i_end = n
    j_end = j+k
    if (j_end > m):
        j_end = m

    # print("\t>>", i, j, i_start, i_end, j_start,  j_end, n, m)
    NumberOfElements = (i_end-i_start + 1) * (j_end-j_start + 1)
    SumOfElements = 0
    for ii in range(i_start, i_end):
        for jj in range(j_start, j_end):
            SumOfElements = SumOfElements+In_Field[ii][jj]

    AverageValue = SumOfElements / NumberOfElements

    ij_value = AverageValue

    return(ij_value)


def Line2Row_AverageThis(InputField, k):
    
    m = len(InputField[0])
    n = len(InputField)

    OutputField = [[0 for x in range(m)] for y in range(n)]

    for i in range(0, n):
        for j in range(0, m):

            OutputField[i][j] = AverageThisPoint(i, j, InputField, k)

    return(OutputField)


def Diagonally_AverageThis(InputField, k):

    n = len(InputField[0])
    m = len(InputField)

    OutputField = [[0 for x in range(n)] for y in range(m)]

    for i_begin in range(0, n):
        i = i_begin
        j = 0
        while i >= 0 and j <= m-1:

            OutputField[j][i] = AverageThisPoint(i, j, InputField, k)

            if (i >= 0):
                i = i-1
            if (j <= m-1):
                j = j+1

    for j_begin in range(1, m):
        j = j_begin
        i = n-1
        while i >= 0 and j <= m-1:

            OutputField[j][i] = AverageThisPoint(i, j, InputField, k)

            if (i >= 0):
                i = i-1
            if (j <= m-1):
                j = j+1

    return (OutputField)

def Test():
    AveragingWidth=1
    w, h = 5, 3
    InputField = [[0 for x in range(w)] for y in range(h)]
    InputField[0][0] = 15
    print(InputField)
    TempField=InputField
    print(Line2Row_AverageThis(TempField, AveragingWidth))

    TempField=InputField
    print(Diagonally_AverageThis(TempField, AveragingWidth))

