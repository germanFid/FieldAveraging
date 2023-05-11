__kernel void average_gauss(__global const float* input_field, __global float* output_field, __constant long long* field_size, __global const float* window, __constant long long* window_size, __constant float* sum)
{
    long long index = get_global_id(0);
    float temp_elem = 0;
    long long start = 0;
    long long temp_index = index - *window_size;
    // if(temp_index < 0)
    // {
    //     start = -temp_index;
    //     temp_index = 0;
    // }
    for(long long window_index = start; window_index < *window_size * 2 + 1; window_index++)
    {
        temp_index = window_index - (long long)*window_size + index + 4; // if add 4 everything will work
        if(temp_index < *field_size)
            temp_elem += input_field[temp_index] * window[window_index];
        temp_index++;
    }
    output_field[index] = temp_elem / *sum;
}
