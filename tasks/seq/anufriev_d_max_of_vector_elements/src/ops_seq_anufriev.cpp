#include "seq/anufriev_d_max_of_vector_elements/include/ops_seq_anufriev.hpp"

#include <limits>

namespace anufriev_d_max_of_vector_elements_seq {

bool VectorMaxSeq::validation() {
  internal_order_test();

  return !taskData->outputs.empty() && taskData->outputs_count[0] == 1;
}

bool VectorMaxSeq::pre_processing() {
  internal_order_test();

  auto* input_ptr = reinterpret_cast<int32_t*>(taskData->inputs[0]);
  input_.resize(taskData->inputs_count[0]);
  std::copy(input_ptr, input_ptr + taskData->inputs_count[0], input_.begin());

  return true;
}

bool VectorMaxSeq::run() {
  internal_order_test();

  if (input_.empty()) {
    return true;
  }

  max_ = input_[0];
  for (int32_t num : input_) {
    if (num > max_) {
      max_ = num;
    }
  }

  return true;
}

bool VectorMaxSeq::post_processing() {
  internal_order_test();

  *reinterpret_cast<int32_t*>(taskData->outputs[0]) = max_;
  return true;
}

}  // namespace anufriev_d_max_of_vector_elements_seq