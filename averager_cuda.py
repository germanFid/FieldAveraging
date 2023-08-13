from numba import cuda


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
                                  block_shape: tuple = (8, 8, 8)):
    """
    Function that presets cuda GPU setting and launch kernel function of basic 3d averaging.
    Takes cuda type DeviceNDArray and returns same type, further parsing to host is needed.

    Args:
        input_array_gpu (cuda.cudadrv.devicearray.DeviceNDArray): field to get averaged
        radius (int): averaging radius around array point
        block_shape (tuple, optional): Number of threads in each block dimension, total number
        shouldn't be more than 512/1024. Defaults to (8, 8, 8).

    Returns:
        cuda.cudadrv.devicearray.DeviceNDArray: peasantly averaged 3d field
    """
    output_array_gpu = cuda.device_array_like(input_array_gpu)
    threads_per_block = block_shape
    blocks_per_grid_x = (input_array_gpu.shape[0] + threads_per_block[0] - 1)\
        // threads_per_block[0]
    blocks_per_grid_y = (input_array_gpu.shape[1] + threads_per_block[1] - 1)\
        // threads_per_block[1]
    blocks_per_grid_z = (input_array_gpu.shape[2] + threads_per_block[2] - 1)\
        // threads_per_block[2]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)
    cuda_kernel_field_average_3d[blocks_per_grid, threads_per_block](
        input_array_gpu, output_array_gpu, radius)
    return output_array_gpu


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
                                  block_shape: tuple = (16, 16)):
    """
    Function that presets cuda GPU settings and launch kernel function of basic 2d averaging.
    Takes cuda type DeviceNDArray and returns same type, further parsing to host is needed.

    Args:
        input_array_gpu (cuda.cudadrv.devicearray.DeviceNDArray): field to get averaged
        radius (int): averaging radius around array point
        block_shape (tuple, optional): Number of threads in each block dimension, total number
        shouldn't be more than 512/1024. Defaults to (16, 16).

    Returns:
        cuda.cudadrv.devicearray.DeviceNDArray: peasantly averaged 3d field
    """
    output_array_gpu = cuda.device_array_like(input_array_gpu)
    threads_per_block = block_shape
    blocks_per_grid_x = (input_array_gpu.shape[0] + threads_per_block[0] - 1)\
        // threads_per_block[0]
    blocks_per_grid_y = (input_array_gpu.shape[1] + threads_per_block[1] - 1)\
        // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    cuda_kernel_field_average_2d[blocks_per_grid, threads_per_block](
        input_array_gpu, output_array_gpu, radius)
    return output_array_gpu
