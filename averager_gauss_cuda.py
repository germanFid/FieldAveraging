import numpy as np
from numba import cuda
from numba import float32, float64, int32
from typing import Tuple


TPB = 16


def init_gauss_window_cuda(sigma: int) -> Tuple[np.ndarray, int, float]:
    """
    initing gauss window
    Args:
        sigma (int): sigma sets the radius of window and
        influence on blur coef in gauss formula
    Returns:
        Tuple[np.ndarray, float]: returning window and sum of elems in window
    """
    window_size = int(np.ceil(3 * sigma))
    window = np.zeros(2 * window_size + 1)

    s2 = 2 * sigma * sigma
    const = np.sqrt(2 * np.pi) * sigma

    window[window_size] = 1
    for i in range(1, window_size + 1):
        window[window_size - i] = window[window_size +
                                         i] = np.exp(- i * i / s2) / const
    window_sum = np.sum(window)

    return window, window_size, window_sum


def get_blocks_per_grid(shapes: int):
    match shapes:
        case 1:
            threadsPerBlock = (TPB,)
            blocksPerGrid_x = int(np.ceil(data_to_handle.shape[0] / threadsPerBlock[0]))
            blocksPerGrid = (blocksPerGrid_x)
        case 2:
            threadsPerBlock = (TPB, TPB)
            blocksPerGrid_x = int(np.ceil(data_to_handle.shape[0] / threadsPerBlock[0]))
            blocksPerGrid_y = int(np.ceil(data_to_handle.shape[1] / threadsPerBlock[1]))
            blocksPerGrid = (blocksPerGrid_x, blocksPerGrid_y)
        case 3:
            threadsPerBlock = (TPB, TPB, TPB)
            blocksPerGrid_x = int(np.ceil(data_to_handle.shape[0] / threadsPerBlock[0]))
            blocksPerGrid_y = int(np.ceil(data_to_handle.shape[1] / threadsPerBlock[1]))
            blocksPerGrid_z = int(np.ceil(data_to_handle.shape[2] / threadsPerBlock[2]))
            blocksPerGrid = (blocksPerGrid_x, blocksPerGrid_y, blocksPerGrid_z)
        case _:
            print("Strange dimension, exitting...")
            exit()

    return blocksPerGrid, threadsPerBlock


def mov_window_and_data_arrs_to_gpu(
        cpu_window: np.ndarray, cpu_data: np.ndarray) -> Tuple:
    gpu_window = cuda.to_device(cpu_window)
    gpu_old_data = cuda.to_device(cpu_data)
    gpu_new_data = cuda.to_device(cpu_data)
    return gpu_window, gpu_old_data, gpu_new_data


def get_shape_data(array: np.ndarray):
    return array.ndim


def average_with_gauss(cpu_data: np.ndarray, sigma: int) -> np.ndarray:
    dimension = get_shape_data(cpu_data)
    blocksPerGrid, threadsPerBlock = get_blocks_per_grid(dimension)
    cpu_window, window_size, window_sum = init_gauss_window_cuda(sigma)
    gpu_window, gpu_old_data, gpu_new_data = mov_window_and_data_arrs_to_gpu(cpu_window, cpu_data)
    gpu1_new_data = cuda.device_array_like(gpu_new_data)
    print(blocksPerGrid, threadsPerBlock)
    match dimension:
        case 1:
            average_1d_by_x_gauss[blocksPerGrid, threadsPerBlock](gpu_old_data, gpu_new_data, gpu_window,
                                                                  window_size, window_sum)
        case 2:
            average_2d_by_x_gauss[blocksPerGrid, threadsPerBlock](gpu_old_data, gpu_new_data, gpu_window,
                                                                  window_size, window_sum)
            average_2d_by_y_gauss[blocksPerGrid, threadsPerBlock](gpu_new_data, gpu_new_data, gpu_window,
                                                                  window_size, window_sum)
        case 3:
            average_3d_by_x_gauss[blocksPerGrid, threadsPerBlock](gpu_old_data, gpu_new_data, gpu_window,
                                                                  window_size, window_sum)
            average_3d_by_y_gauss[blocksPerGrid, threadsPerBlock](gpu_new_data, gpu_new_data, gpu_window,
                                                                  window_size, window_sum)
            average_3d_by_z_gauss[blocksPerGrid, threadsPerBlock](gpu_new_data, gpu_new_data, gpu_window,
                                                                  window_size, window_sum)
        case _:
            print("Strange dimension, exitting...")
            exit()
    return gpu_new_data.copy_to_host()


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



if (__name__ == '__main__'):
    new_data = average_with_gauss(data_to_handle, 1)
    new_data1 = average_2d_by_gauss(data_to_handle, 1)
    

    (new_data == new_data1).all