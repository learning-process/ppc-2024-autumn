#include "seq/shuravina_o_contrast/include/ops_seq.hpp"

#include <algorithm>
#include <limits>

namespace shuravina_o_contrast {

bool ContrastTaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<uint8_t>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  output_ = std::vector<uint8_t>(taskData->inputs_count[0]);
  min_val_ = *std::min_element(input_.begin(), input_.end());
  max_val_ = *std::max_element(input_.begin(), input_.end());
  return true;
}

bool ContrastTaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool ContrastTaskSequential::run() {
  internal_order_test();
  if (max_val_ == min_val_) {
    std::fill(output_.begin(), output_.end(), 128);
  } else {
    for (size_t i = 0; i < input_.size(); ++i) {
      output_[i] = static_cast<uint8_t>((input_[i] - min_val_) * 255.0 / (max_val_ - min_val_));
    }
  }
  return true;
}

bool ContrastTaskSequential::post_processing() {
  internal_order_test();
  auto* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->outputs[0]);
  for (unsigned i = 0; i < taskData->outputs_count[0]; i++) {
    tmp_ptr[i] = output_[i];
  }
  return true;
}

}  // namespace shuravina_o_contrast