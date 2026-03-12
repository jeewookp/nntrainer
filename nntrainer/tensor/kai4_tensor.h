// SPDX-License-Identifier: Apache-2.0
/**
 * @file	kai4_tensor.h
 * @date	12 January 2026
 * @brief	This is Kai4Tensor class for Kai library integration.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Hyeonggwon <hyeonggwon@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __KAI4_TENSOR_H__
#define __KAI4_TENSOR_H__

#ifdef __cplusplus

#include <tensor_base.h>
#include <vector>


namespace nntrainer {

/**
 * @class Kai4Tensor class
 * @brief Kai4Tensor class for Kai library packed tensors
 * @note  This tensor type uses Kai library's packing format (NxK, RHS packing)
 */
class Kai4Tensor : public TensorBase {

public:
  /**
   * @brief     Basic Constructor of Tensor
   */
  Kai4Tensor(std::string name_ = "", Tformat fm = Tformat::NCHW, QScheme qscheme_ = QScheme::PER_CHANNEL_AFFINE);

  /**
   * @brief Construct a new Kai4Tensor object
   *
   * @param d Tensor dim for this tensor
   * @param alloc_now Allocate memory to this tensor or not
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   * @param qscheme_ Quantization scheme (default PER_BLOCK_AFFINE for Kai)
   */
  Kai4Tensor(const TensorDim &d, bool alloc_now,
              Initializer init = Initializer::NONE, std::string name = "",
              QScheme qscheme_ = QScheme::PER_CHANNEL_AFFINE, unsigned int idx_variant = 5);

  /**
   * @brief Construct a new Kai4Tensor object
   *
   * @param d Tensor dim for this tensor
   * @param buf buffer
   */
  Kai4Tensor(const TensorDim &d, const void *buf = nullptr);

  /**
   * @brief Construct a new Kai4Tensor object
   * @param rhs TensorBase object to copy
   */
  Kai4Tensor(TensorBase &rhs) : TensorBase(rhs) {}

  /**
   * @copydoc Tensor::allocate()
   */
  void allocate() override;

  /**
   * @copydoc Tensor::deallocate()
   */
  void deallocate() override {
    data = nullptr;
    offset = 0;
  }

  /**
   * @copydoc Tensor::getData()
   */
  void *getData() const override;

  /**
   * @copydoc Tensor::getData()
   */
  void *getData(size_t idx) const override {
    throw std::invalid_argument(
      "Kai4Tensor::getData(idx) is not supported. Use getData() instead.");
  }

  /**
   * @copydoc Tensor::getAddress()
   */
  void *getAddress(unsigned int i) override {
    throw std::invalid_argument("Kai4Tensor::getAddress() is not supported.");
  }

  /**
   * @copydoc Tensor::getAddress()
   */
  const void *getAddress(unsigned int i) const override {
    throw std::invalid_argument("Kai4Tensor::getAddress() is not supported.");
  }

  /**
   * @copydoc Tensor::setValue()
   */
  void setValue(float value) override {
    throw std::invalid_argument("Kai4Tensor::setValue() is not supported.");
  }

  /**
   * @copydoc Tensor::setValue()
   */
  void setValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value) override {
    throw std::invalid_argument("Kai4Tensor::setValue() is not supported.");
  }

  /**
   * @copydoc Tensor::addValue()
   */
  void addValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value, float beta) override {
    throw std::invalid_argument("Kai4Tensor::addValue() is not supported.");
  }

  /**
   * @copydoc Tensor::setZero()
   */
  void setZero() override;

  /**
   * @copydoc Tensor::initialize()
   */
  void initialize(Initializer init) override {
    throw std::invalid_argument("Kai4Tensor::initialize(init) is not supported.");
  }

  /**
   * @copydoc Tensor::initialize()
   */
  void initialize() override;

  /**
   * @copydoc Tensor::print()
   */
  void print(std::ostream &out) const override {
    out << "Kai4Tensor: " << getName() << " Dim: " << getDim();
  }

  /**
   * @copydoc Tensor::copy()
   */
  void copy(const Tensor &from) override {
    throw std::invalid_argument("Kai4Tensor::copy() is not supported. Use pack() instead.");
  }

  /**
   * @copydoc Tensor::copyData()
   */
  void copyData(const Tensor &from) override {
    throw std::invalid_argument("Kai4Tensor::copyData() is not supported.");
  }

  /**
   * @copydoc Tensor::copy_with_stride()
   */
  void copy_with_stride(const Tensor &input, Tensor &output) override {
    throw std::invalid_argument(
      "Kai4Tensor::copy_with_stride() is not supported.");
  }

  /**
   * @copydoc Tensor::max_abs()
   */
  float max_abs() const override {
    throw std::invalid_argument("Kai4Tensor::max_abs() is not supported.");
  }

  /**
   * @copydoc Tensor::maxValue()
   */
  float maxValue() const override {
    throw std::invalid_argument("Kai4Tensor::maxValue() is not supported.");
  }

  /**
   * @copydoc Tensor::minValue()
   */
  float minValue() const override {
    throw std::invalid_argument("Kai4Tensor::minValue() is not supported.");
  }

  /**
   * @copydoc TensorBase::size()
   */
  size_t size() const override;

  /**
   * @copydoc Tensor::getMemoryBytes()
   */
  size_t getMemoryBytes() const override;

  /**
   * @copydoc Tensor::q_scheme()
   */
  QScheme q_scheme() const override;

  
  void pack(const uint8_t *rhs, const float *bias, size_t nr, size_t kr,
            size_t sr, size_t bl = 32);

  /**
   * @brief Set the packing parameters (needed before allocation if not strictly determined by dim)
   *
   * @param nr Number of columns written by the matmul micro-kernel
   * @param kr Number of columns loaded in the innermost loop
   * @param bl Block length
   */

  void setKernelVariant(uint32_t variant_idx);

  /**
   * @brief Get the current kernel variant index
   * @return Current kernel variant index
   */
  uint32_t getKernelVariant() const { return _idx_variant; }


private:
  /**
   * @brief quantization scheme
   */
  QScheme qscheme;

  /**
   * @brief copy a buffer to @a this, the caller has to ensure that @a this is
   * initialized otherwise undefined behavior
   * @note Adapting copy_q40 style
   *
   * @param buf buffer to copy from (must be pre-packed data if copying directly,
   *            or use pack() for raw data)
   */
  void copy_kai(const void *buf);

  /**
   * @brief  Get the Data Type String object
   * @return std::string of tensor data type
   */
  std::string getStringDataType() const override { return "QInt4_KAI"; }

  /**
   * @copydoc Tensor::isValid()
   */
  bool isValid() const override { return true; }

  
  
  // Kernel variant index (default to variant 6: 8x8x32 GEMM dotprod for wider hardware support)
  // dotprod (ARMv8.2-A) is more widely available than i8mm (ARMv8.6-A)
  uint32_t _idx_variant;

  // Trans or Not
  // Set to True because kxn format is not supported currently (Date : 2026/01/19)
  bool transB = true;
  
 
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __KAI4_TENSOR_H__ */
