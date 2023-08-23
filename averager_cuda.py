from CUDA_general import np, cuda
from CUDA_general import VCardLaunchData, GaussWindowData


def cuda_basic_averaging(input_array: np.ndarray, radius: int, iterations: int = 1) -> np.ndarray:
    launch_data = VCardLaunchData(input_array)
    input_array_gpu = cuda.to_device(input_array)

    match input_array.ndim:
        case 2:
            output = cuda_basic_2d_array_averaging(input_array_gpu, radius,
                                                   launch_data,
                                                   iterations)
        case 3:
            output = cuda_basic_3d_array_averaging(input_array_gpu, radius,
                                                   launch_data,
                                                   iterations)
        case _:
            print("Strange dimension, exitting...")
            exit()
    return output


def cuda_basic_3d_array_averaging(input_array_gpu: cuda.cudadrv.devicearray.DeviceNDArray,
                                  radius: int,
                                  launch_data: VCardLaunchData,
                                  iterations: int = 1):
    """
    Function that presets cuda GPU setting and launch kernel function of basic 3d averaging.
    Takes cuda type DeviceNDArray and returns same type, further parsing to host is needed.

    Args:
        input_array_gpu (cuda.cudadrv.devicearray.DeviceNDArray): field to get averaged
        radius (int): averaging radius around array point
        launch_data (VCardLaunchData): Contains threads per block and block per grid info

    Returns:
        cuda.cudadrv.devicearray.DeviceNDArray: peasantly averaged 3d field
    """
    output_array_gpu = cuda.device_array(input_array_gpu.shape, input_array_gpu.dtype)
    for i in range(iterations):
        cuda_kernel_field_average_3d[launch_data.blocksPerGrid, launch_data.threadsPerBlock](
            input_array_gpu, output_array_gpu, radius)
        input_array_gpu = output_array_gpu
    return output_array_gpu.copy_to_host()


@cuda.jit('void(float64[:, :, :], float64[:, :, :], int32)')
def cuda_kernel_field_average_3d(input_array: cuda.cudadrv.devicearray.DeviceNDArray,
                                 output_array: cuda.cudadrv.devicearray.DeviceNDArray,
                                 radius: int):
    """
    Cuda GPU Kernel function that use absolute gpu indexing of threads to go through
    given array and and use basic 3d averaging method on each point.

    Args:
        input_array (cuda.cudadrv.devicearray.DeviceNDArray): field to get averaged
        output_array (cuda.cudadrv.devicearray.DeviceNDArray): peasantly averaged 3d field
        radius (int): averaging radius around array point
    """
    i, j, k = cuda.grid(3)
    if i < output_array.shape[0] and j < output_array.shape[1] and k < output_array.shape[2]:
        output_array[i, j, k] = cuda_average_this_3d_point(input_array, radius, i, j, k)


@cuda.jit('float64(float64[:, :, :], int32, int32, int32, int32)', device=True)
def cuda_average_this_3d_point(input_array: cuda.cudadrv.devicearray.DeviceNDArray,
                               radius: int, i: int, j: int, k: int):
    """
    Function that is executed on each thread of cuda GPU. Takes average value of
    all point around given point with given radius.

    Args:
        input_array (cuda.cudadrv.devicearray.DeviceNDArray): field to get average value from
        radius (int): averaging radius around array point
        i (int): index in row
        j (int): index in column
        k (int): index in depth

    Returns:
        (float64): peasantly averaged value of our 3d point in field
    """
    sum_val = 0.0
    count = 0
    for radius_i in range(-radius, radius + 1):
        for radius_j in range(-radius, radius + 1):
            for radius_k in range(-radius, radius + 1):
                i_radius = i + radius_i
                j_radius = j + radius_j
                k_radius = k + radius_k
                i_in_array = i_radius >= 0 and i_radius < input_array.shape[0]
                j_in_array = j_radius >= 0 and j_radius < input_array.shape[1]
                k_in_array = k_radius >= 0 and k_radius < input_array.shape[2]
                if i_in_array and j_in_array and k_in_array:
                    sum_val += input_array[i_radius, j_radius, k_radius]
                    count += 1
    return sum_val / count


def cuda_basic_2d_array_averaging(input_array_gpu: cuda.cudadrv.devicearray.DeviceNDArray,
                                  radius: int,
                                  launch_data: VCardLaunchData,
                                  iterations: int = 1):
    """
    Function that presets cuda GPU setting and launch kernel function of basic 2d averaging.
    Takes cuda type DeviceNDArray and returns same type, further parsing to host is needed.

    Args:
        input_array_gpu (cuda.cudadrv.devicearray.DeviceNDArray): field to get averaged
        radius (int): averaging radius around array point
        launch_data (VCardLaunchData): Contains threads per block and block per grid info

    Returns:
        cuda.cudadrv.devicearray.DeviceNDArray: peasantly averaged 2d field
    """
    output_array_gpu = cuda.device_array(input_array_gpu.shape, input_array_gpu.dtype)
    for i in range(iterations):
        cuda_kernel_field_average_2d[launch_data.blocksPerGrid, launch_data.threadsPerBlock](
            input_array_gpu, output_array_gpu, radius)
        input_array_gpu = output_array_gpu
    return output_array_gpu.copy_to_host()


@cuda.jit('void(float64[:, :], float64[:, :], int32)')
def cuda_kernel_field_average_2d(input_array: cuda.cudadrv.devicearray.DeviceNDArray,
                                 output_array: cuda.cudadrv.devicearray.DeviceNDArray,
                                 radius: int):
    """
    Cuda GPU Kernel function that use absolute GPU indexing of threads to go through
    given array and and use basic 2d averaging method on each point.

    Args:
        input_array (cuda.cudadrv.devicearray.DeviceNDArray): field to get averaged
        output_array (cuda.cudadrv.devicearray.DeviceNDArray): peasantly averaged 2d field
        radius (int): averaging radius around array point
    """
    i, j = cuda.grid(2)
    if i < output_array.shape[0] and j < output_array.shape[1]:
        output_array[i, j] = cuda_average_this_2d_point(input_array, radius, i, j)


@cuda.jit('float64(float64[:, :], int32, int32, int32)', device=True)
def cuda_average_this_2d_point(input_array: cuda.cudadrv.devicearray.DeviceNDArray,
                               radius: int, i: int, j: int):
    """
    Function that is executed on each thread of cuda GPU. Takes average value of
    all point around given point with given radius.

    Args:
        input_array (cuda.cudadrv.devicearray.DeviceNDArray): field to get average value from
        radius (int): averaging radius around array point
        i (int): index in row
        j (int): index in column

    Returns:
        (float64): peasantly averaged value of our 2d point in field
    """
    sum_val = 0.0
    count = 0
    for radius_i in range(-radius, radius + 1):
        for radius_j in range(-radius, radius + 1):
            i_radius = i + radius_i
            j_radius = j + radius_j
            i_in_array = i_radius >= 0 and i_radius < input_array.shape[0]
            j_in_array = j_radius >= 0 and j_radius < input_array.shape[1]
            if i_in_array and j_in_array:
                sum_val += input_array[i_radius, j_radius]
                count += 1
    return sum_val / count


def average_with_gauss(cpu_data: np.ndarray, sigma: int, iterations: int = 1) -> np.ndarray:
    launch_data = VCardLaunchData(cpu_data)
    window_data = GaussWindowData(sigma)
    window_data.transfer_window_to_gpu()
    gpu_old_data = cuda.to_device(cpu_data)

    match cpu_data.ndim:
        case 1:
            output = average_1d_by_gauss(gpu_old_data, launch_data, window_data, iterations)
        case 2:
            output = average_2d_by_gauss(gpu_old_data, launch_data, window_data, iterations)
        case 3:
            output = average_3d_by_gauss(gpu_old_data, launch_data, window_data, iterations)
        case _:
            print("Strange dimension, exitting...")
            exit()
    return output


def average_1d_by_gauss(old_data: cuda.cudadrv.devicearray.DeviceNDArray,
                        launch_data: VCardLaunchData, window_data: GaussWindowData,
                        iterations: int = 1):
    oper_data = cuda.device_array(old_data)
    for i in range(iterations):
        if i % 2 == 0:
            average_1d_by_x_gauss[launch_data.blocksPerGrid,
                                  launch_data.threadsPerBlock](old_data, oper_data,
                                                               window_data.gpu_window,
                                                               window_data.size,
                                                               window_data.sum)
        else:
            average_1d_by_x_gauss[launch_data.blocksPerGrid,
                                  launch_data.threadsPerBlock](oper_data, old_data,
                                                               window_data.gpu_window,
                                                               window_data.size,
                                                               window_data.sum)
    if iterations % 2 == 1:
        old_data = oper_data
    return old_data.copy_to_host()


def average_2d_by_gauss(old_data: cuda.cudadrv.devicearray.DeviceNDArray,
                        launch_data: VCardLaunchData, window_data: GaussWindowData,
                        iterations: int = 1):
    oper_data = cuda.device_array(old_data)
    for i in range(iterations):
        average_2d_by_x_gauss[launch_data.blocksPerGrid,
                              launch_data.threadsPerBlock](old_data, oper_data,
                                                           window_data.gpu_window, window_data.size,
                                                           window_data.sum)
        average_2d_by_y_gauss[launch_data.blocksPerGrid,
                              launch_data.threadsPerBlock](oper_data, old_data,
                                                           window_data.gpu_window, window_data.size,
                                                           window_data.sum)
    return old_data.copy_to_host()


def average_3d_by_gauss(old_data: cuda.cudadrv.devicearray.DeviceNDArray,
                        launch_data: VCardLaunchData, window_data: GaussWindowData,
                        iterations: int = 1):

    oper_data = cuda.device_array_like(old_data)
    oper_data2 = cuda.device_array_like(old_data)
    for i in range(iterations):
        average_3d_by_x_gauss[launch_data.blocksPerGrid,
                              launch_data.threadsPerBlock](old_data, oper_data,
                                                           window_data.gpu_window, window_data.size,
                                                           window_data.sum)
        average_3d_by_y_gauss[launch_data.blocksPerGrid,
                              launch_data.threadsPerBlock](oper_data, oper_data2,
                                                           window_data.gpu_window, window_data.size,
                                                           window_data.sum)
        average_3d_by_z_gauss[launch_data.blocksPerGrid,
                              launch_data.threadsPerBlock](oper_data2, old_data,
                                                           window_data.gpu_window, window_data.size,
                                                           window_data.sum)
    return old_data.copy_to_host()


@cuda.jit('void(float64[:], float64[:], float64[:],\
           int32, float64)')
def average_1d_by_x_gauss(old_data: cuda.cudadrv.devicearray.DeviceNDArray,
                          new_data: cuda.cudadrv.devicearray.DeviceNDArray,
                          window: cuda.cudadrv.devicearray.DeviceNDArray,
                          window_size: int, window_sum: float):
    x_th = cuda.grid(1)

    if (x_th < old_data.shape[0]):
        temp_elem = 0
        temp_index = x_th - window_size
        window_index = 0
        if temp_index < 0:
            window_index = window_size - x_th
            temp_index = 0

        while (window_index < window_size * 2 + 1 and temp_index < old_data.shape[0]):
            temp_elem = temp_elem + old_data[temp_index] * window[window_index]

            temp_index += 1
            window_index += 1

        new_data[x_th] = temp_elem / window_sum


@cuda.jit('void(float64[:, :], float64[:, :], float64[:],\
           int32, float64)')
def average_2d_by_x_gauss(old_data: cuda.cudadrv.devicearray.DeviceNDArray,
                          new_data: cuda.cudadrv.devicearray.DeviceNDArray,
                          window: cuda.cudadrv.devicearray.DeviceNDArray,
                          window_size: int, window_sum: float):
    y_th, x_th = cuda.grid(2)

    if (y_th < old_data.shape[0] and x_th < old_data.shape[1]):
        temp_elem = 0
        temp_index = x_th - window_size
        window_index = 0
        if temp_index < 0:
            window_index = window_size - x_th
            temp_index = 0

        while (window_index < window_size * 2 + 1 and temp_index < old_data.shape[1]):
            temp_elem += old_data[y_th][temp_index] * window[window_index]

            temp_index += 1
            window_index += 1

        new_data[y_th][x_th] = temp_elem / window_sum


@cuda.jit('void(float64[:, :], float64[:, :], float64[:],\
           int32, float64)')
def average_2d_by_y_gauss(old_data: cuda.cudadrv.devicearray.DeviceNDArray,
                          new_data: cuda.cudadrv.devicearray.DeviceNDArray,
                          window: cuda.cudadrv.devicearray.DeviceNDArray,
                          window_size: int, window_sum: float):
    y_th, x_th = cuda.grid(2)

    if (y_th < old_data.shape[0] and x_th < old_data.shape[1]):
        temp_elem = 0
        temp_index = y_th - window_size
        window_index = 0
        if temp_index < 0:
            window_index = window_size - y_th
            temp_index = 0

        while (window_index < window_size * 2 + 1 and temp_index < old_data.shape[0]):
            temp_elem += old_data[temp_index][x_th] * window[window_index]

            temp_index += 1
            window_index += 1

        new_data[y_th][x_th] = temp_elem / window_sum


@cuda.jit('void(float64[:, :, :], float64[:, :, :], float64[:],\
           int32, float64)')
def average_3d_by_x_gauss(old_data: cuda.cudadrv.devicearray.DeviceNDArray,
                          new_data: cuda.cudadrv.devicearray.DeviceNDArray,
                          window: cuda.cudadrv.devicearray.DeviceNDArray,
                          window_size: int, window_sum: float):
    z_th, y_th, x_th = cuda.grid(3)

    if (z_th < old_data.shape[0] and y_th < old_data.shape[1] and x_th < old_data.shape[2]):
        temp_elem = 0
        temp_index = x_th - window_size
        window_index = 0
        if temp_index < 0:
            window_index = window_size - x_th
            temp_index = 0

        while (window_index < window_size * 2 + 1 and temp_index < old_data.shape[2]):
            temp_elem += old_data[z_th][y_th][temp_index] * window[window_index]

            temp_index += 1
            window_index += 1

        new_data[z_th][y_th][x_th] = temp_elem / window_sum


@cuda.jit('void(float64[:, :, :], float64[:, :, :], float64[:],\
           int32, float64)')
def average_3d_by_y_gauss(old_data: cuda.cudadrv.devicearray.DeviceNDArray,
                          new_data: cuda.cudadrv.devicearray.DeviceNDArray,
                          window: cuda.cudadrv.devicearray.DeviceNDArray,
                          window_size: int, window_sum: float):
    z_th, y_th, x_th = cuda.grid(3)

    if (z_th < old_data.shape[0] and y_th < old_data.shape[1] and x_th < old_data.shape[2]):
        temp_elem = 0
        temp_index = y_th - window_size
        window_index = 0
        if temp_index < 0:
            window_index = window_size - y_th
            temp_index = 0

        while (window_index < window_size * 2 + 1 and temp_index < old_data.shape[1]):
            temp_elem += old_data[z_th][temp_index][x_th] * window[window_index]

            temp_index += 1
            window_index += 1

        new_data[z_th][y_th][x_th] = temp_elem / window_sum


@cuda.jit('void(float64[:, :, :], float64[:, :, :], float64[:],\
           int32, float64)')
def average_3d_by_z_gauss(old_data: cuda.cudadrv.devicearray.DeviceNDArray,
                          new_data: cuda.cudadrv.devicearray.DeviceNDArray,
                          window: cuda.cudadrv.devicearray.DeviceNDArray,
                          window_size: int, window_sum: float):
    z_th, y_th, x_th = cuda.grid(3)

    if (z_th < old_data.shape[0] and y_th < old_data.shape[1] and x_th < old_data.shape[2]):
        temp_elem = 0
        temp_index = z_th - window_size
        window_index = 0
        if temp_index < 0:
            window_index = window_size - z_th
            temp_index = 0

        while (window_index < window_size * 2 + 1 and temp_index < old_data.shape[0]):
            temp_elem += old_data[temp_index][y_th][x_th] * window[window_index]

            temp_index += 1
            window_index += 1

        new_data[z_th][y_th][x_th] = temp_elem / window_sum
