#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))


// Reads FP32 image, writes FP16 transposed image
// Fuses FP32->FP16 conversion with transpose
__attribute__((qcom_reqd_sub_group_size("full"))) kernel void
input_transpose(__read_only image1d_buffer_t input, __write_only image1d_buffer_t input_transposed,
                              const int alignK_4, const int M_4) {
    const int col_id = get_global_id(0);
    const int row_id = get_global_id(1);
    const int col_id_4 = col_id<<2;
    const int row_id_4 = row_id<<2;

    float4 f0 = read_imagef(input, alignK_4 * row_id_4 + col_id);
    float4 f1 = read_imagef(input, alignK_4 * (row_id_4+1) + col_id);
    float4 f2 = read_imagef(input, alignK_4 * (row_id_4+2) + col_id);
    float4 f3 = read_imagef(input, alignK_4 * (row_id_4+3) + col_id);

    half4 data0 = convert_half4(f0);
    half4 data1 = convert_half4(f1);
    half4 data2 = convert_half4(f2);
    half4 data3 = convert_half4(f3);

    write_imageh(input_transposed, M_4 * col_id_4 + row_id, (half4)(data0.s0,data1.s0,data2.s0,data3.s0));
    write_imageh(input_transposed, M_4 * (col_id_4+1) + row_id, (half4)(data0.s1,data1.s1,data2.s1,data3.s1));
    write_imageh(input_transposed, M_4 * (col_id_4+2) + row_id, (half4)(data0.s2,data1.s2,data2.s2,data3.s2));
    write_imageh(input_transposed, M_4 * (col_id_4+3) + row_id, (half4)(data0.s3,data1.s3,data2.s3,data3.s3));
}
