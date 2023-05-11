import pyopencl as cl
import pyopencl.array as pycl_array
import numpy as np


def init_gauss_window(sigma: int) -> np.ndarray:

    window_size = int(np.ceil(3 * sigma))
    window = np.zeros(2 * window_size + 1)

    s2 = 2 * sigma * sigma
    const = np.sqrt(2 * np.pi) * sigma

    window[window_size] = 1
    for i in range(1, window_size + 1):
        window[window_size - i] = window[window_size +
                                         i] = np.exp(- i * i / s2) / const

    return window


def get_info_about_devices():
    for platform in cl.get_platforms():
        print('=' * 60)
        print('Platform - Name:  ' + platform.name)
        print('Platform - Vendor:  ' + platform.vendor)
        print('Platform - Version:  ' + platform.version)
        print('Platform - Profile:  ' + platform.profile)
        # Print each device per-platform
        for device in platform.get_devices():
            print('    ' + '-' * 56)
            print('    Device - Name:  ' + device.name)
            print('    Device - Type:  ' + cl.device_type.to_string(device.type))
            print(
                '    Device - Max Clock Speed:  {0} Mhz'.format(device.max_clock_frequency))
            print(
                '    Device - Compute Units:  {0}'.format(device.max_compute_units))
            print(
                '    Device - Local Memory:  {0:.0f} KB'.format(device.local_mem_size / 1024.0))
            print('    Device - Constant Memory:  {0:.0f} KB'.format(
                device.max_constant_buffer_size / 1024.0))
            print(
                '    Device - Global Memory: {0:.0f} GB'.format(device.global_mem_size / 1073741824.0))
            print('    Device - Max Buffer/Image Size: {0:.0f} MB'.format(
                device.max_mem_alloc_size / 1048576.0))
            print(
                '    Device - Max Work Group Size: {0:.0f}'.format(device.max_work_group_size))
    print('\n')


def get_device():
    device = cl.get_platforms()[0].get_devices()[-1]
    print(device)
    return device


def get_context():
    return cl.Context([get_device()])


# context = get_context()
# queue = cl.CommandQueue(context)

platform = cl.get_platforms()[0]  # Select the first platform [0]
device = platform.get_devices()[-1]  # Select the first device on this platform [0]
context = cl.Context([device])  # Create a context with your device
queue = cl.CommandQueue(context)  # Create a command queue with your context

sigma = 1
window = np.array(init_gauss_window(sigma)).astype(np.float32)

in_field = np.array([0, 15, 0, 15, 12, 0]).astype(np.float32)
print(in_field)
out_field = np.empty_like(in_field)

window_sum = np.array(float(np.sum(window)))
window_size = np.size(window)
in_field_size = np.size(in_field)
wnd_sum_arr = np.array(window_sum).astype(np.float32)

cl_buff_input = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=in_field)
cl_buff_window = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=window)
cl_buff_output = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, out_field.nbytes)
cl_buff_field_size = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(in_field_size))
cl_buff_window_sum = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=wnd_sum_arr)
cl_buff_window_size = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(window_size))

print("field size : ", in_field_size)
gauss_code = open('univer/2_course/OPD/FieldAveraging/pure_code_cl.c', 'r').read()

gauss_program = cl.Program(context, gauss_code).build()
gauss_program.average_gauss(queue, in_field.shape, None,
                            cl_buff_input, cl_buff_output, cl_buff_field_size,
                            cl_buff_window, cl_buff_window_size, cl_buff_window_sum)

cl.enqueue_copy(queue, out_field, cl_buff_output)
for i in out_field:
    print(i)
