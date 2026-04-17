MAIN_FUNCTION($0) {
  int DST_X = GLOBAL_ID_0;
  int DST_Y = GLOBAL_ID_1;
  int DST_S = GLOBAL_ID_2;
  DST_X *= 2;
  DST_Y *= 2;
  DST_S *= 2;
  if (DST_S >= args.dst_tensor_slices) return;
  if (DST_X >= args.dst_tensor_width || DST_Y >= args.dst_tensor_height || DST_S >= args.dst_tensor_slices) {
    return;
  }
  ACCUM_FLT4 r_w0_h0_s0 = INIT_ACCUM_FLT4(0.0f);
  ACCUM_FLT4 r_w1_h0_s0 = INIT_ACCUM_FLT4(0.0f);
  ACCUM_FLT4 r_w0_h1_s0 = INIT_ACCUM_FLT4(0.0f);
  ACCUM_FLT4 r_w1_h1_s0 = INIT_ACCUM_FLT4(0.0f);
  ACCUM_FLT4 r_w0_h0_s1 = INIT_ACCUM_FLT4(0.0f);
  ACCUM_FLT4 r_w1_h0_s1 = INIT_ACCUM_FLT4(0.0f);
  ACCUM_FLT4 r_w0_h1_s1 = INIT_ACCUM_FLT4(0.0f);
  ACCUM_FLT4 r_w1_h1_s1 = INIT_ACCUM_FLT4(0.0f);
  int xc0 = DST_X + 0;
  int xc1 = DST_X + 1;
  int yc0 = DST_Y + 0;
  int yc1 = DST_Y + 1;
  int addr_w0_h0 = (((0) * args.src_tensor_height + (yc0)) * args.src_tensor_width + (xc0));
  int addr_w1_h0 = (((0) * args.src_tensor_height + (yc0)) * args.src_tensor_width + (xc1));
  int addr_w0_h1 = (((0) * args.src_tensor_height + (yc1)) * args.src_tensor_width + (xc0));
  int addr_w1_h1 = (((0) * args.src_tensor_height + (yc1)) * args.src_tensor_width + (xc1));
  int ds = args.src_tensor_slice_stride;
  int s = 0;
  do {
    half4 src_w0_h0;
    half4 src_w1_h0;
    half4 src_w0_h1;
    half4 src_w1_h1;
    FLT4 f0 = read_imageh(args.weights0_image2d, smp_zero, (int2)((DST_S + 0), (s)));
    FLT4 f1 = read_imageh(args.weights1_image2d, smp_zero, (int2)((DST_S + 0), (s)));
    FLT4 f2 = read_imageh(args.weights2_image2d, smp_zero, (int2)((DST_S + 0), (s)));
    FLT4 f3 = read_imageh(args.weights3_image2d, smp_zero, (int2)((DST_S + 0), (s)));
    FLT4 f4 = read_imageh(args.weights0_image2d, smp_zero, (int2)((DST_S + 1), (s)));
    FLT4 f5 = read_imageh(args.weights1_image2d, smp_zero, (int2)((DST_S + 1), (s)));
    FLT4 f6 = read_imageh(args.weights2_image2d, smp_zero, (int2)((DST_S + 1), (s)));
    FLT4 f7 = read_imageh(args.weights3_image2d, smp_zero, (int2)((DST_S + 1), (s)));
    src_w0_h0 = read_imageh(args.src_tensor_image_buffer, addr_w0_h0);
    addr_w0_h0 += ds;
    src_w1_h0 = read_imageh(args.src_tensor_image_buffer, addr_w1_h0);
    addr_w1_h0 += ds;
    src_w0_h1 = read_imageh(args.src_tensor_image_buffer, addr_w0_h1);
    addr_w0_h1 += ds;
    src_w1_h1 = read_imageh(args.src_tensor_image_buffer, addr_w1_h1);
    addr_w1_h1 += ds;
    s += 1;
    r_w0_h0_s0 += f0 * src_w0_h0.x;
    r_w1_h0_s0 += f0 * src_w1_h0.x;
    r_w0_h1_s0 += f0 * src_w0_h1.x;
    r_w1_h1_s0 += f0 * src_w1_h1.x;
    r_w0_h0_s0 += f1 * src_w0_h0.y;
    r_w1_h0_s0 += f1 * src_w1_h0.y;
    r_w0_h1_s0 += f1 * src_w0_h1.y;
    r_w1_h1_s0 += f1 * src_w1_h1.y;
    r_w0_h0_s0 += f2 * src_w0_h0.z;
    r_w1_h0_s0 += f2 * src_w1_h0.z;
    r_w0_h1_s0 += f2 * src_w0_h1.z;
    r_w1_h1_s0 += f2 * src_w1_h1.z;
    r_w0_h0_s0 += f3 * src_w0_h0.w;
    r_w1_h0_s0 += f3 * src_w1_h0.w;
    r_w0_h1_s0 += f3 * src_w0_h1.w;
    r_w1_h1_s0 += f3 * src_w1_h1.w;
    r_w0_h0_s1 += f4 * src_w0_h0.x;
    r_w1_h0_s1 += f4 * src_w1_h0.x;
    r_w0_h1_s1 += f4 * src_w0_h1.x;
    r_w1_h1_s1 += f4 * src_w1_h1.x;
    r_w0_h0_s1 += f5 * src_w0_h0.y;
    r_w1_h0_s1 += f5 * src_w1_h0.y;
    r_w0_h1_s1 += f5 * src_w0_h1.y;
    r_w1_h1_s1 += f5 * src_w1_h1.y;
    r_w0_h0_s1 += f6 * src_w0_h0.z;
    r_w1_h0_s1 += f6 * src_w1_h0.z;
    r_w0_h1_s1 += f6 * src_w0_h1.z;
    r_w1_h1_s1 += f6 * src_w1_h1.z;
    r_w0_h0_s1 += f7 * src_w0_h0.w;
    r_w1_h0_s1 += f7 * src_w1_h0.w;
    r_w0_h1_s1 += f7 * src_w0_h1.w;
    r_w1_h1_s1 += f7 * src_w1_h1.w;
  } while (s < args.src_tensor_slices);
  if (DST_S + 0 >= args.dst_tensor_slices) return;
  {
    FLT4 bias_val = args.biases_buffer[DST_S + 0];
  {
    FLT4 res = TO_FLT4(r_w0_h0_s0) + bias_val;
    args.dst_tensor_buffer[(((DST_S + 0) * args.dst_tensor_height + (DST_Y + 0)) * args.dst_tensor_width + (DST_X + 0))] = res;
  }
  if (DST_X + 1 < args.dst_tensor_width) {
    FLT4 res = TO_FLT4(r_w1_h0_s0) + bias_val;
    args.dst_tensor_buffer[(((DST_S + 0) * args.dst_tensor_height + (DST_Y + 0)) * args.dst_tensor_width + (DST_X + 1))] = res;
  }
  if (DST_Y + 1 < args.dst_tensor_height) {
    FLT4 res = TO_FLT4(r_w0_h1_s0) + bias_val;
    args.dst_tensor_buffer[(((DST_S + 0) * args.dst_tensor_height + (DST_Y + 1)) * args.dst_tensor_width + (DST_X + 0))] = res;
  }
  if (DST_X + 1 < args.dst_tensor_width && DST_Y + 1 < args.dst_tensor_height) {
    FLT4 res = TO_FLT4(r_w1_h1_s0) + bias_val;
    args.dst_tensor_buffer[(((DST_S + 0) * args.dst_tensor_height + (DST_Y + 1)) * args.dst_tensor_width + (DST_X + 1))] = res;
  }
  }
  if (DST_S + 1 >= args.dst_tensor_slices) return;
  {
    FLT4 bias_val = args.biases_buffer[DST_S + 1];
  {
    FLT4 res = TO_FLT4(r_w0_h0_s1) + bias_val;
    args.dst_tensor_buffer[(((DST_S + 1) * args.dst_tensor_height + (DST_Y + 0)) * args.dst_tensor_width + (DST_X + 0))] = res;
  }
  if (DST_X + 1 < args.dst_tensor_width) {
    FLT4 res = TO_FLT4(r_w1_h0_s1) + bias_val;
    args.dst_tensor_buffer[(((DST_S + 1) * args.dst_tensor_height + (DST_Y + 0)) * args.dst_tensor_width + (DST_X + 1))] = res;
  }
  if (DST_Y + 1 < args.dst_tensor_height) {
    FLT4 res = TO_FLT4(r_w0_h1_s1) + bias_val;
    args.dst_tensor_buffer[(((DST_S + 1) * args.dst_tensor_height + (DST_Y + 1)) * args.dst_tensor_width + (DST_X + 0))] = res;
  }
  if (DST_X + 1 < args.dst_tensor_width && DST_Y + 1 < args.dst_tensor_height) {
    FLT4 res = TO_FLT4(r_w1_h1_s1) + bias_val;
    args.dst_tensor_buffer[(((DST_S + 1) * args.dst_tensor_height + (DST_Y + 1)) * args.dst_tensor_width + (DST_X + 1))] = res;
  }
  }
}
