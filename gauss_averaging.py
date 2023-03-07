import math

MAX_HOR = 15
MAX_VER = 3
sigma = 1

mtx = [0] * MAX_VER
t_mtx = [0] * MAX_VER
for i in range(MAX_VER):
    mtx[i] = [0] * MAX_HOR
    t_mtx[i] = [0] * MAX_HOR

mtx[0][0] = 5
mtx[MAX_VER - 1][MAX_HOR - 1] = 5
for j in range(MAX_VER):
    for i in range(MAX_HOR):
        print(mtx[j][i], end = "  ")
    print()
print()

wnd_mid = wnd_max_sz = 150
window = [0] * (2 * wnd_max_sz + 1)
tmp_hor = [0] * MAX_HOR
tmp_ver = [0] * MAX_VER

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

# hor averagin first
for j in range(MAX_VER):
    for i in range(MAX_HOR):
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
            t_ind = i + k
                
            if(t_ind >= 0 and t_ind < MAX_HOR):
                t_elem += mtx[j][t_ind] * window[k + wnd_mid]
                sum += window[k + wnd_mid]
        
        tmp_hor[i] = float(t_elem) / sum
            
    
    for t in range(MAX_HOR):
        t_mtx[j][t] = tmp_hor[t]

# important to copy after we've changed mtx
mtx = t_mtx.copy()

# ver aver sec

for i in range(MAX_HOR):
    for j in range(MAX_VER):
        sum = 0
        t_elem = 0

        for k in range(-wnd_sz, wnd_sz + 1):
            t_ind = j + k

            if(t_ind >= 0 and t_ind < MAX_VER):
                t_elem += mtx[t_ind][i] * window[k + wnd_mid]
                sum += window[k + wnd_mid]
        
        tmp_ver[j] = float(t_elem) / sum
    
    for t in range(MAX_VER):
        t_mtx[t][i] = tmp_ver[t]

# important to copy after we've changed mtx
mtx = t_mtx.copy()

sum = 0
for j in range(MAX_VER):
    for i in range(MAX_HOR):
        sum += mtx[j][i]    
        print('%.2f' % mtx[j][i], end = "  ")
    print()
print(sum)
        
            


    

