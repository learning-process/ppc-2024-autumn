#include "seq/shuravina_o_contrast/include/ops_seq.hpp"

#include <algorithm>

namespace shuravina_o_contrast {

bool ContrastTaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<uint8_t>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  output_ = std::vector<uint8_t>(taskData->inputs_count[0]);
  return true;
}

bool ContrastTaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool ContrastTaskSequential::run() {
  internal_order_test();
  uint8_t min_val = *std::min_element(input_.begin(), input_.end());
  uint8_t max_val = *std::max_element(input_.begin(), input_.end());

  for (size_t i = 0; i < input_.size(); ++i) {
    output_[i] = static_cast<uint8_t>((input_[i] - min_val) * 255.0 / (max_val - min_val));
  }
  return true;
}

bool ContrastTaskSequential::post_processing() {
  internal_order_test();
  auto* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->outputs[0]);
  std::copy(output_.begin(), output_.end(), tmp_ptr);
  return true;
}

}  // namespace shuravina_o_contrast