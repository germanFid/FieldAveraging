from typing import Tuple
from numba import cuda
import numpy as np
import os


TPB = 8


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


def detect_cuda():
    """
    Detect supported CUDA hardware and print a summary of the detected hardware.

    Returns a number of supported devices that were detected.
    """
    devlist = cuda.list_devices()
    print('Found %d CUDA devices' % len(devlist))
    supported_count = 0
    for dev in devlist:
        attrs = []
        cc = dev.compute_capability
        kernel_timeout = dev.KERNEL_EXEC_TIMEOUT
        tcc = dev.TCC_DRIVER
        fp32_to_fp64_ratio = dev.SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO
        attrs += [('Compute Capability', '%d.%d' % cc)]
        attrs += [('PCI Device ID', dev.PCI_DEVICE_ID)]
        attrs += [('PCI Bus ID', dev.PCI_BUS_ID)]
        attrs += [('UUID', dev.uuid)]
        attrs += [('Watchdog', 'Enabled' if kernel_timeout else 'Disabled')]
        if os.name == "nt":
            attrs += [('Compute Mode', 'TCC' if tcc else 'WDDM')]
        attrs += [('FP32/FP64 Performance Ratio', fp32_to_fp64_ratio)]
        if cc < (3, 5):
            support = '[NOT SUPPORTED: CC < 3.5]'
        elif cc < (5, 0):
            support = '[SUPPORTED (DEPRECATED)]'
            supported_count += 1
        else:
            support = '[SUPPORTED]'
            supported_count += 1

        print('id %d    %20s %40s' % (dev.id, dev.name, support))
        for key, val in attrs:
            print('%40s: %s' % (key, val))

    print('Summary:')
    print('\t%d/%d devices are supported' % (supported_count, len(devlist)))
    return supported_count
