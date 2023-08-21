from numba import cuda
import numpy as np

from CUDA_general import VCardLaunchData
from averager_gauss_cuda import cuda_gauss_averaging


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
