import numpy as np
from numba import cuda
from numba import float32, float64, int32
from typing import Tuple


TPB = 16


class GaussWindowData:

    def __init__(self, sigma):
        self.sigma = sigma
        self.window, self.size, self.sum = self.init_gauss_window_cuda(sigma)
        self.gpu_window = []

    def init_gauss_window_cuda(self, sigma: int) -> Tuple[np.ndarray, int, float]:
        """
        initing gauss window
        Args:
            sigma (int): sigma sets the radius of window and
            influence on blur coef in gauss formula
        Returns:
            Tuple[np.ndarray, int, float]: returning window, window size and 
                                           sum of elems in window
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

    def transfer_window_to_gpu(self):
        self.gpu_window = cuda.to_device(self.window)


class VCardLaunchData:
    def __init__(self, data_to_handle: np.ndarray):
        self.blocksPerGrid, self.threadsPerBlock = self.get_blocks_threads_per_grid(data_to_handle)

    def get_blocks_threads_per_grid(self, data_to_handle: np.ndarray):
        match data_to_handle.ndim:
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


def cuda_gauss_averaging(cpu_data: np.ndarray, sigma: int, iterations: int = 1) -> np.ndarray:
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
