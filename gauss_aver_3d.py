import math

MAX_DEP = 5
MAX_VER = 3
MAX_HOR = 15

sigma = 5

mtx = [0] * MAX_DEP
t_mtx = [0] * MAX_DEP

for i in range(MAX_DEP):
    mtx[i] = [0] * MAX_VER
    t_mtx[i] = [0] * MAX_DEP

for j in range(MAX_DEP):
    for i in range(MAX_VER):
        mtx[j][i] = [0] * MAX_HOR
        t_mtx[j][i] = [0] * MAX_HOR
                
        
mtx[0][MAX_VER - 1][MAX_HOR - 1] = 1500

# printing only the first (zero) layer
for j in range(MAX_VER):
    for i in range(MAX_HOR):
        print('%4d' % mtx[0][j][i], end = "  ")
    print()
print()

wnd_mid = wnd_max_sz = 150
window = [0] * (2 * wnd_max_sz + 1)
tmp_hor = [0] * MAX_HOR
tmp_ver = [0] * MAX_VER
tmp_dep = [0] * MAX_DEP


s2 = 2 * sigma * sigma
wnd_sz = math.ceil(3 * sigma)

if(wnd_sz > wnd_max_sz):
    print("Too big sigma")
    exit()

# middle elem is e^0 == 1
window[wnd_mid] = 1

# wnd init
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
                    t_elem += mtx[z][y][t_ind] * window[k + wnd_mid]
                    sum += window[k + wnd_mid]
            
            tmp_hor[x] = t_elem / sum
                
        
        for t in range(MAX_HOR):
            t_mtx[z][y][t] = tmp_hor[t]

    # important to copy after we've changed mtx
    mtx = t_mtx.copy()


    # ver aver sec
    for x in range(MAX_HOR):
        for y in range(MAX_VER):
            sum = 0
            t_elem = 0

            for k in range(-wnd_sz, wnd_sz + 1):
                t_ind = y + k

                if(t_ind >= 0 and t_ind < MAX_VER):
                    t_elem += mtx[z][t_ind][x] * window[k + wnd_mid]
                    sum += window[k + wnd_mid]
            
            tmp_ver[y] = t_elem / sum
        
        for t in range(MAX_VER):
            t_mtx[z][t][x] = tmp_ver[t]

    mtx = t_mtx.copy()

    # depth aver
    for x in range(MAX_HOR):
        for y in range(MAX_VER):
            sum = 0
            t_elem = 0

            for k in range(-wnd_sz, wnd_sz + 1):
                t_ind = z + k

                if(t_ind >= 0 and t_ind < MAX_VER):
                    t_elem += mtx[t_ind][y][x] * window[k + wnd_mid]
                    sum += window[k + wnd_mid]
            
            tmp_dep[z] = t_elem / sum
        
        for t in range(MAX_DEP):
            t_mtx[t][y][x] = tmp_dep[t]

    mtx = t_mtx.copy()

# testing output
sum = 0
for z in range(MAX_DEP):
    for y in range(MAX_VER):
        for x in range(MAX_HOR):
            sum += mtx[z][y][x]    
            # print('%.2f' % mtx[y][x], end = "  ")
        # print()
print(sum)