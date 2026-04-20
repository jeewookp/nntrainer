// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Thummala Pallavi <t.pallavi@samsung.com>
 *
 * @file        rmsnorm_layer_cl.cpp
 * @date        8 June 2024
 * @brief       This is RMSNorm Layer Class for Neural Network with
 * OpenCl implementation
 * @see         https://github.com/nntrainer/nntrainer
 * @author      Thummala Pallavi <t.pallavi@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 */

#include <blas_kernels.h>
#include <cl_kernels/image_reformat.h>
#include <cl_kernels/rmsnorm.h>
#ifdef ENABLE_FP16
#include <cl_kernels/rmsnorm_fp16.h>
#include <cl_kernels/rmsnorm_image2d.h>
#endif
#include <common_properties.h>
#include <gpu_image_pool.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <rmsnorm_layer_cl.h>
#include <util_func.h>
#include <CL/cl.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum RMSParams { gamma };

RMSNormLayerCl::RMSNormLayerCl() : LayerImplCl() { wt_idx.fill(0); }

bool RMSNormLayerCl::registerClKernels(ClContext &cl_context) {
  auto &layer_kernel_ptrs = getLayerKernelPtrs();

  // check if already registered
  if (!layer_kernel_ptrs.empty()) {
    ml_loge("kernels for reshape layer are already registered");
    return false;
  }

  do {

    ClContext::SharedPtrClKernel kernel_rmsnorm_ptr =
      cl_context.registerClKernel(rmsnorm_kernel, "rmsnorm_cl");
    if (!kernel_rmsnorm_ptr) {
      ml_loge("OpenCL Error: Fail to register rmsnorm_cl kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_rmsnorm_ptr);

#ifdef ENABLE_FP16
    ClContext::SharedPtrClKernel kernel_rmsnorm_fp16_ptr =
      cl_context.registerClKernel(rmsnorm_fp16_kernel, "rmsnorm_cl_fp16");
    if (!kernel_rmsnorm_fp16_ptr) {
      ml_loge("OpenCL Error: Fail to register rmsnorm_cl_fp16 kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_rmsnorm_ptr);

    ClContext::SharedPtrClKernel kernel_rmsnorm_image2d_ptr =
      cl_context.registerClKernel(rmsnorm_image2d_kernel, "rmsnorm_image2d");
    if (!kernel_rmsnorm_image2d_ptr) {
      ml_loge("OpenCL Error: Fail to register rmsnorm_image2d kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_rmsnorm_image2d_ptr);
#endif

    return true;

  } while (false);

  // clear all registered kernels if any error occurs during registration
  // layer_kernel_ptrs.clear();

  return false;
}

void RMSNormLayerCl::finalize(InitLayerContext &context) {
  std::vector<TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);
  auto &rmsparams_gamma = std::get<props::GammaInitializer>(rmsnorm_props);

  TensorDim gamma_dim(
    1, 1, 1, dim[0].width(),
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()));
  wt_idx[RMSParams::gamma] =
    context.requestWeight(gamma_dim, rmsparams_gamma, WeightRegularizer::NONE,
                          1.0f, 0.0f, "gamma", false);
}

void RMSNormLayerCl::forwarding(RunLayerContext &context, bool training) {
  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);
  auto &epsilon = std::get<props::Epsilon>(rmsnorm_props).get();
  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    rmsnormProcess(in, out, gamma, epsilon);
  } else {
#ifdef ENABLE_FP16
    rmsnormProcess_fp16(in, out, gamma, epsilon);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void RMSNormLayerCl::rmsnormProcess(Tensor const &input, Tensor &result,
                                    Tensor const &gamma, const float epsilon) {
  rmsnorm_cl(input.getData<float>(), gamma.getData<float>(),
             result.getData<float>(), epsilon,
             input.batch() * input.channel() * input.height(), input.width());
}

#ifdef ENABLE_FP16
void RMSNormLayerCl::rmsnormProcess_fp16(Tensor const &input, Tensor &result,
                                         Tensor const &gamma,
                                         const float epsilon) {

  bool ret = false;
  int dim1 = input.batch() * input.height() * input.width() * input.channel();
  CREATE_IF_EMPTY_DIMS(result, input.batch(), input.channel(), input.height(),
                       input.width(), input.getTensorType());
  int b = input.batch();
  int c = input.channel();
  int h = input.height();
  int w = input.width();

  auto *global_cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  do {
    const _FP16 *data = input.getData<_FP16>();
    _FP16 *rdata = result.getData<_FP16>();
    const _FP16 *gdata = gamma.getData<_FP16>();

    // Phase A image2d path: if the input tensor has an image2d waiting
    // for it in GpuImagePool (written by an upstream image2d-aware
    // layer) OR the tensor is SVM-backed and we can svm_to_image2d
    // ourselves, run rmsnorm_image2d and publish the output image2d
    // to the pool. Downstream delegate gemms pick it up and skip
    // their svm_to_image2d reformat.
    //
    // Shape assumption: RMSNorm's width W is the channel dimension the
    // kernel normalises over, total positions M = B*C*H, slice count
    // = W/4. Matches how delegate gemm indexes its src_img.
    const int rms_M = b * c * h;
    const int rms_slices = w / 4;
    const bool image2d_ok =
      (w % 4) == 0 && input.getMemoryData() &&
      input.getMemoryData()->isSVM() && result.getMemoryData() &&
      result.getMemoryData()->isSVM();

    if (image2d_ok) {
      auto kernel_image2d_ptr =
        getLayerKernelPtrs()[Kernels::RMSNORM_IMAGE2D];
      cl_context clctx = global_cl_context->context_inst_.GetContext();
      cl_int err = 0;

      // One-shot src/dst image2d cache, re-sized when shape changes.
      static cl_mem s_src_img = nullptr;
      static cl_mem s_dst_img = nullptr;
      static int s_cached_M = 0, s_cached_slices = 0;
      cl_image_format fmt_h = {CL_RGBA, CL_HALF_FLOAT};
      if (!s_src_img || s_cached_M != rms_M ||
          s_cached_slices != rms_slices) {
        if (s_src_img) clReleaseMemObject(s_src_img);
        if (s_dst_img) clReleaseMemObject(s_dst_img);
        cl_image_desc d = {};
        d.image_type = CL_MEM_OBJECT_IMAGE2D;
        d.image_width = rms_M; d.image_height = rms_slices;
        s_src_img = clCreateImage(clctx, CL_MEM_READ_WRITE, &fmt_h, &d, 0, &err);
        s_dst_img = clCreateImage(clctx, CL_MEM_READ_WRITE, &fmt_h, &d, 0, &err);
        s_cached_M = rms_M; s_cached_slices = rms_slices;
      }

      // Source: pool hit reuses upstream's image2d; else pull SVM -> s_src_img.
      cl_mem src_img_use = s_src_img;
      int pool_w = 0, pool_h_ = 0;
      cl_mem pooled =
        GpuImagePool::Global().get(data, &pool_w, &pool_h_);
      if (pooled && pool_w == rms_M && pool_h_ == rms_slices) {
        src_img_use = pooled;
      } else {
        // Use the delegate's shared svm_to_image2d kernel — the kernel
        // source for image_reformat.cl is compiled and registered at
        // the gemm path; re-register locally for this layer's use.
        static ClContext::SharedPtrClKernel s_in_kern;
        if (!s_in_kern) {
          s_in_kern = global_cl_context->registerClKernel(
            image_reformat_kernel, "svm_to_image2d");
        }
        int a = 0;
        s_in_kern->SetKernelSVMArguments(a++, const_cast<_FP16 *>(data));
        s_in_kern->SetKernelArguments(a++, &s_src_img, sizeof(cl_mem));
        int sm = rms_M, sk = w;
        s_in_kern->SetKernelArguments(a++, &sm, sizeof(int));
        s_in_kern->SetKernelArguments(a++, &sk, sizeof(int));
        const int ig[3] = {(int)((rms_M+15)/16)*16,
                            (int)((rms_slices+15)/16)*16, 1};
        const int il[3] = {16, 16, 1};
        global_cl_context->command_queue_inst_.DispatchCommand(
          s_in_kern, ig, il);
      }

      // rmsnorm_image2d kernel: (src_img, dst_img, gamma_svm, epsilon,
      //                          M, slices)
      int a = 0;
      kernel_image2d_ptr->SetKernelArguments(a++, &src_img_use, sizeof(cl_mem));
      kernel_image2d_ptr->SetKernelArguments(a++, &s_dst_img, sizeof(cl_mem));
      kernel_image2d_ptr->SetKernelSVMArguments(
        a++, const_cast<_FP16 *>(gdata));
      kernel_image2d_ptr->SetKernelArguments(a++, &epsilon, sizeof(cl_half));
      int mm = rms_M, ss = rms_slices;
      kernel_image2d_ptr->SetKernelArguments(a++, &mm, sizeof(int));
      kernel_image2d_ptr->SetKernelArguments(a++, &ss, sizeof(int));
      const int wg_count[3] = {(int)((rms_M+31)/32)*32, 1, 1};
      const int wg_size[3]  = {32, 1, 1};
      ret = global_cl_context->command_queue_inst_.DispatchCommand(
        kernel_image2d_ptr, wg_count, wg_size);
      if (!ret) break;

      // Register the output image2d so the downstream gemm_delegate
      // can skip its svm_to_image2d reformat.
      GpuImagePool::Global().set(rdata, s_dst_img, rms_M, rms_slices);

      // Also image2d_to_svm into the tensor's SVM region so any layer
      // that doesn't yet understand the pool still sees a valid
      // output. If later phases convert every downstream consumer
      // to image2d, this can be dropped entirely.
      static ClContext::SharedPtrClKernel s_out_kern;
      if (!s_out_kern) {
        s_out_kern = global_cl_context->registerClKernel(
          image_reformat_kernel, "image2d_to_svm");
      }
      a = 0;
      s_out_kern->SetKernelArguments(a++, &s_dst_img, sizeof(cl_mem));
      s_out_kern->SetKernelSVMArguments(a++, rdata);
      int om = rms_M, on = w;
      s_out_kern->SetKernelArguments(a++, &om, sizeof(int));
      s_out_kern->SetKernelArguments(a++, &on, sizeof(int));
      const int og[3] = {(int)((rms_M+15)/16)*16,
                          (int)((rms_slices+15)/16)*16, 1};
      const int ol[3] = {16, 16, 1};
      global_cl_context->command_queue_inst_.DispatchCommand(
        s_out_kern, og, ol);
      break;
    }

    auto kernel_rmsnorm_ptr = getLayerKernelPtrs()[Kernels::RMSNORM_CL_FP16];

    // Ensure the host view of `data` is coherent before WriteDataRegion
    // reads it. The upstream gemm_delegate_fp16_cl only enqueues a
    // non-blocking SVMMap so its GPU writes to this tensor's SVM region
    // may still be in flight, and clEnqueueWriteBuffer(CL_TRUE,
    // host_ptr=data) reads host memory synchronously at the call site
    // rather than at queue execution time. Drain the queue here with a
    // blocking SVMMap(CL_MAP_READ) so the subsequent WriteDataRegion
    // copies coherent data into the cl_mem scratch buffer.
    if (input.getMemoryData() && input.getMemoryData()->isSVM()) {
      global_cl_context->command_queue_inst_.enqueueSVMMap(
        const_cast<_FP16 *>(data), dim1 * sizeof(_FP16),
        /*read_only=*/true);
    }

    ret = clbuffInstance.getInBufferA()->WriteDataRegion(
      global_cl_context->command_queue_inst_, dim1 * sizeof(cl_half), data);
    if (!ret) break;

    ret = clbuffInstance.getInBufferB()->WriteDataRegion(
      global_cl_context->command_queue_inst_, input.width() * sizeof(cl_half),
      gdata);
    if (!ret) break;

    ret = kernel_rmsnorm_ptr->SetKernelArguments(
      0, clbuffInstance.getInBufferA()->GetBuffer(), sizeof(cl_mem));
    if (!ret) break;
    ret = kernel_rmsnorm_ptr->SetKernelArguments(
      1, clbuffInstance.getOutBufferA()->GetBuffer(), sizeof(cl_mem));
    if (!ret) break;
    ret = kernel_rmsnorm_ptr->SetKernelArguments(
      2, clbuffInstance.getInBufferB()->GetBuffer(), sizeof(cl_mem));
    if (!ret) break;
    ret = kernel_rmsnorm_ptr->SetKernelArguments(3, &epsilon, sizeof(cl_half));
    if (!ret) break;
    ret = kernel_rmsnorm_ptr->SetKernelArguments(4, &b, sizeof(int));
    if (!ret) break;
    ret = kernel_rmsnorm_ptr->SetKernelArguments(5, &c, sizeof(int));
    if (!ret) break;
    ret = kernel_rmsnorm_ptr->SetKernelArguments(6, &h, sizeof(int));
    if (!ret) break;
    ret = kernel_rmsnorm_ptr->SetKernelArguments(7, &w, sizeof(int));
    if (!ret) break;

    const int work_groups_count[3] = {b * c, h, 1};
    /// @todo: create a group size by device & input
    const int work_group_size[3] = {w, 1, 1}; // test-value

    ret = global_cl_context->command_queue_inst_.DispatchCommand(
      kernel_rmsnorm_ptr, work_groups_count, work_group_size);
    if (!ret) break;

    ret = clbuffInstance.getOutBufferA()->ReadDataRegion(
      global_cl_context->command_queue_inst_, dim1 * sizeof(cl_half), rdata);
    if (!ret) break;
  } while (false);
}
#endif

void RMSNormLayerCl::incremental_forwarding(nntrainer::RunLayerContext &context,
                                            unsigned int from, unsigned int to,
                                            bool training) {
  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);
  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }

  in_step_dim.height(to - from);
  out_step_dim.height(to - from);

  Tensor in_step = in.getSharedDataTensor(in_step_dim, 0, true);
  Tensor out_step = out.getSharedDataTensor(out_step_dim, 0, true);

  auto &epsilon = std::get<props::Epsilon>(rmsnorm_props).get();

  if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
    rmsnormProcess(in, out, gamma, epsilon);
  } else {
#ifdef ENABLE_FP16
    rmsnormProcess_fp16(in, out, gamma, epsilon);
#else
    throw std::runtime_error("enable-fp16 is not enabled");
#endif
  }
}

void RMSNormLayerCl::calcDerivative(RunLayerContext &context) {
  ml_logi("Training not supported");
}

void RMSNormLayerCl::calcGradient(RunLayerContext &context) {
  ml_logi("Training not supported");
}

void RMSNormLayerCl::exportTo(Exporter &exporter,
                              const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(rmsnorm_props, method, this);
}

void RMSNormLayerCl::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, rmsnorm_props);
  LayerImpl::setProperty(remain_props);
}

std::vector<ClContext::SharedPtrClKernel> &
RMSNormLayerCl::getLayerKernelPtrs() {
  /**< kernel list relevant with this layer */
  static std::vector<ClContext::SharedPtrClKernel> layer_kernel_ptrs;
  return layer_kernel_ptrs;
}

} // namespace nntrainer
