#include "seq/milovankin_m_sum_of_vector_elements/include/ops_seq.hpp"

namespace milovankin_m_sum_of_vector_elements_seq {

bool VectorSumSeq::validation() {
  internal_order_test();

  return !taskData->outputs.empty() && taskData->outputs_count[0] == 1;
}

bool VectorSumSeq::pre_processing() {
  internal_order_test();

  // Fill input vector from taskData
  auto* input_ptr = reinterpret_cast<int32_t*>(taskData->inputs[0]);
  input_.resize(taskData->inputs_count[0]);
  std::copy(input_ptr, input_ptr + taskData->inputs_count[0], input_.begin());

  return true;
}

bool VectorSumSeq::run() {
  internal_order_test();

  sum_ = 0;
  for (int32_t num : input_) {
    sum_ += num;
  }

  return true;
}

bool VectorSumSeq::post_processing() {
  internal_order_test();
  *reinterpret_cast<int64_t*>(taskData->outputs[0]) = sum_;
  return true;
}

}  // namespace milovankin_m_sum_of_vector_elements_seq
