def SquareAveraging(OurField, i, j, k):

    m = len(OurField[0])
    n = len(OurField)

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
            SumOfElements = SumOfElements+OurField[ii][jj]

    AverageValue = SumOfElements / NumberOfElements

    ij_value = AverageValue

    return (ij_value)


def AverageThisField(OurField, k):
    print("AverageThisField")
    m = len(OurField[0])
    n = len(OurField)

    OutputField = [[0 for x in range(m)] for y in range(n)]

    for i in range(0, n):
        for j in range(0, m):

            print("ElementNumber = "+str(i)+", "+str(j))

            OutputField[i][j] = SquareAveraging(OurField, i, j, k)

    return (OutputField)


def DiagonalAverageThisField(OurField, k):

    print("DiagonalAverageThisField")
    n = len(OurField[0])
    m = len(OurField)

    print(">> n, m", n, m)

    OutputField = [[0 for x in range(n)] for y in range(m)]

    for i_begin in range(0, n):
        i = i_begin
        j = 0
        while i >= 0 and j <= m-1:

            print("ElementNumber = " + str(i) + ", " + str(j))

            OutputField[j][i] = SquareAveraging(OurField, i, j, k)

            if (i >= 0):
                i = i-1
            if (j <= m-1):
                j = j+1

    for j_begin in range(1, m):
        j = j_begin
        i = n-1
        while i >= 0 and j <= m-1:

            print("ElementNumber = "+str(i)+", "+str(j))

            OutputField[j][i] = SquareAveraging(OurField, i, j, k)

            if (i >= 0):
                i = i-1
            if (j <= m-1):
                j = j+1

    return (OutputField)


def FieldAveraging(InputArray):

    k = 1

    print(InputArray)

    for i in range(1, 3):
        print("GlobalIterNumber = "+str(i))
        TempArray = AverageThisField(InputArray, k)
        InputArray = TempArray
        print(TempArray)

    for i in range(1, 3):
        print("GlobalIterNumber = "+str(i))
        TempArray = DiagonalAverageThisField(InputArray, k)
        

    return TempArray


def Test():
    w, h = 5, 3
    InputArray = [[0 for x in range(w)] for y in range(h)]
    InputArray[0][0] = 15
    print(InputArray)
    print(FieldAveraging(InputArray))


# KEKW()
