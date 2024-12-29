
#include "seq/vershinina_a_image_smoothing/include/ops_seq.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool vershinina_a_image_smoothing::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_ = reinterpret_cast<int*>(taskData->inputs[0]);
  rows = taskData->inputs_count[0];
  cols = taskData->inputs_count[1];
  output_.resize(rows * cols);
  return true;
}

bool vershinina_a_image_smoothing::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs.size() == 1 && taskData->inputs_count.size() == 2 && taskData->outputs.size() == 1;
}

bool vershinina_a_image_smoothing::TestTaskSequential::run() {
  internal_order_test();
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      int sum = 0;
      int c = 0;
      for (int row = std::max(0, i - 1); row <= std::min(i + 1, rows - 1); row++) {
        for (int col = std::max(0, j - 1); col <= std::min(j + 1, cols - 1); col++) {
          sum += input_[row * cols + col];
          c++;
        }
      }
      output_[i * cols + j] = sum / c;
    }
  }
  return true;
}

bool vershinina_a_image_smoothing::TestTaskSequential::post_processing() {
  internal_order_test();
  std::copy(output_.begin(), output_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  return true;
}
