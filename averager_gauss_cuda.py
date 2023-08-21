import numpy as np
from numba import cuda
from CUDA_general import VCardLaunchData, GaussWindowData


def average_with_gauss(cpu_data: np.ndarray, sigma: int, iterations: int = 1) -> np.ndarray:
    launch_data = VCardLaunchData(cpu_data)
    window_data = GaussWindowData(sigma)
    window_data.transfer_window_to_gpu()
    gpu_old_data = cuda.to_device(cpu_data)

    match cpu_data.ndim:
        case 1:
            output = average_1d_by_gauss(gpu_old_data, launch_data, window_data)
        case 2:
            output = average_2d_by_gauss(gpu_old_data, launch_data, window_data)
        case 3:
            output = average_3d_by_gauss(gpu_old_data, launch_data, window_data)
        case _:
            print("Strange dimension, exitting...")
            exit()
    return output


def average_1d_by_gauss(old_data: cuda.cudadrv.devicearray.DeviceNDArray,
                        launch_data: VCardLaunchData, window_data: GaussWindowData,
                        iterations: int = 1):
    oper_data = cuda.device_array_like(old_data)
    oper2_data = cuda.device_array(old_data)
    for i in range(iterations):
        if i % 2 == 0:
            average_1d_by_x_gauss[launch_data.blocksPerGrid,
                                  launch_data.threadsPerBlock](oper_data, oper2_data,
                                                               window_data.gpu_window,
                                                               window_data.size,
                                                               window_data.sum)           
        else:
            average_1d_by_x_gauss[launch_data.blocksPerGrid,
                                  launch_data.threadsPerBlock](oper2_data, oper_data,
                                                               window_data.gpu_window,
                                                               window_data.size,
                                                               window_data.sum)
    if iterations % 2 == 1:
        return oper2_data.copy_to_host()
    return oper_data.copy_to_host()



def average_2d_by_gauss(old_data: cuda.cudadrv.devicearray.DeviceNDArray,
                        launch_data: VCardLaunchData, window_data: GaussWindowData,
                        iterations: int = 1):
    output = cuda.device_array_like(old_data)
    oper_data = cuda.device_array(old_data)
    for i in range(iterations):
        average_2d_by_x_gauss[launch_data.blocksPerGrid,
                              launch_data.threadsPerBlock](output, oper_data,
                                                           window_data.gpu_window, window_data.size,
                                                           window_data.sum)
        average_2d_by_y_gauss[launch_data.blocksPerGrid,
                              launch_data.threadsPerBlock](oper_data, output,
                                                           window_data.gpu_window, window_data.size,
                                                           window_data.sum)
    return output.copy_to_host()


def average_3d_by_gauss(old_data: cuda.cudadrv.devicearray.DeviceNDArray,
                        launch_data: VCardLaunchData, window_data: GaussWindowData,
                        iterations: int = 1):
    output = cuda.device_array(old_data)
    oper_data = cuda.device_array_like(old_data)
    for i in range(iterations):
        average_3d_by_x_gauss[launch_data.blocksPerGrid,
                              launch_data.threadsPerBlock](oper_data, output,
                                                           window_data.gpu_window, window_data.size,
                                                           window_data.sum)
        average_3d_by_y_gauss[launch_data.blocksPerGrid,
                              launch_data.threadsPerBlock](output, oper_data,
                                                           window_data.gpu_window, window_data.size,
                                                           window_data.sum)
        average_3d_by_z_gauss[launch_data.blocksPerGrid,
                              launch_data.threadsPerBlock](oper_data, output,
                                                           window_data.gpu_window, window_data.size,
                                                           window_data.sum)
    return output.copy_to_host()


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
