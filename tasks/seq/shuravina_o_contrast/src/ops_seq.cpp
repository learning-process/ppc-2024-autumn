#include "seq/shuravina_o_contrast/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

bool shuravina_o_contrast::ContrastSequential::pre_processing() {
  input_ = std::vector<uint8_t>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  output_ = std::vector<uint8_t>(taskData->outputs_count[0]);
  return true;
}

bool shuravina_o_contrast::ContrastSequential::validation() {
  return taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool shuravina_o_contrast::ContrastSequential::run() {
  uint8_t min_val = *std::min_element(input_.begin(), input_.end());
  uint8_t max_val = *std::max_element(input_.begin(), input_.end());

  std::cout << "Min: " << static_cast<int>(min_val) << ", Max: " << static_cast<int>(max_val) << std::endl;

  if (min_val == max_val) {
    std::fill(output_.begin(), output_.end(), 255);
  } else {
    for (size_t i = 0; i < input_.size(); ++i) {
      output_[i] = static_cast<uint8_t>((input_[i] - min_val) * 255.0 / (max_val - min_val));
      std::cout << "Output[" << i << "]: " << static_cast<int>(output_[i]) << std::endl;
    }
  }
  return true;
}

bool shuravina_o_contrast::ContrastSequential::post_processing() {
  auto* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->outputs[0]);
  for (unsigned i = 0; i < taskData->outputs_count[0]; i++) {
    tmp_ptr[i] = output_[i];
  }
  return true;
}