#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__attribute__((qcom_reqd_sub_group_size("full"))) kernel void
repack_kai_to_adreno(const __global uchar* kai_packed_data,
    __global ushort* weights,
    __global half* scales,
    const int N, const int K, const int rhs_packed_stride, const int quantization_group_size) {

    const int k_id = get_global_id(0);
    const int n_id = get_global_id(1);
    vload4(0,kai_packed_data + n_id * rhs_packed_stride + k_id * 64);
    if ((k_id==0)&&(n_id==0)){
        
    }
}
