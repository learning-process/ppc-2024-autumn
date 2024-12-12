#include <algorithm>

#include "seq/shuravina_o_contrast/include/ops_seq.hpp"

bool shuravina_o_contrast::ContrastTaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<uint8_t>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  output_ = std::vector<uint8_t>(taskData->inputs_count[0]);
  return true;
}

bool shuravina_o_contrast::ContrastTaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool shuravina_o_contrast::ContrastTaskSequential::run() {
  internal_order_test();
  uint8_t min_val = *std::min_element(input_.begin(), input_.end());
  uint8_t max_val = *std::max_element(input_.begin(), input_.end());

  for (size_t i = 0; i < input_.size(); ++i) {
    output_[i] = static_cast<uint8_t>((input_[i] - min_val) * 255.0 / (max_val - min_val));
  }
  return true;
}

bool shuravina_o_contrast::ContrastTaskSequential::post_processing() {
  internal_order_test();
  auto* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->outputs[0]);
  for (unsigned i = 0; i < taskData->outputs_count[0]; i++) {
    tmp_ptr[i] = output_[i];
  }
  return true;
}