#include "seq/sharamygina_i_horizontal_line_filtration/include/ops_seq.h"

bool sharamygina_i_horizontal_line_filtration_seq::horizontal_line_filtration_seq::pre_processing() {
  internal_order_test();

  rows_ = taskData->inputs_count[0];
  cols_ = taskData->inputs_count[1];

  if (world.rank() == 0) {
    auto* input_buffer = reinterpret_cast<unsigned char*>(taskData->inputs[0]);
    original_data_.assign(input_buffer, input_buffer + rows_ * cols_);
  }

  return true;
}

bool sharamygina_i_horizontal_line_filtration_seq::horizontal_line_filtration_seq::validation() {
  internal_order_test();

    if (!taskData || taskData->inputs.size() < 9 || taskData->inputs_count.size() < 2 || taskData->outputs.empty() ||
        taskData->outputs_count.empty()) {
      return false;
    }
  

  int exp_size = taskData->inputs_count[0] * taskData->inputs_count[1];
  if (taskData->inputs_count[0] < 3 || taskData->inputs_count[1] < 3 || taskData->inputs_count[0] < world.size() ||
      taskData->inputs.size != exp_size || taskData->outputs.size != exp_size) {
    return false;
  }
  return true;
}

bool sharamygina_i_horizontal_line_filtration_seq::horizontal_line_filtration_seq::run() {
  internal_order_test();
  for (int i = 0; i < rows_; i++)
    for (int j = 0; j < cols_; j++) result_data_[i * cols + j] = InputAnotherPixel(origin_data_, i, j, rows_, cols_);

  return true;
}

bool sharamygina_i_horizontal_line_filtration_seq::horizontal_line_filtration_seq::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int* output_ptr = reinterpret_cast<unsigned char*>(taskData->outputs[0]);
    std::copy(processed_data_.begin(), processed_data_.end(), output_ptr);
  }
  return true;
}

unsigned int sharamygina_i_horizontal_line_filtration_seq::horizontal_line_filtration_seq::InputAnotherPixel(
    const std::vector<unsigned int>& image, int x, int y, int rows, int cols) {
  unsigned int sum = 0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      int tX = x + i - 1, tY = y + j - 1;
      if (tX < 0 || tX > rows_ - 1) tX = x;
      if (tY < 0 || tY > cols_ - 1) tY = y;
      if (tX * cols + tY >= cols * rows) {
        tX = x;
        tY = y;
      }
      sum += static_cast<unsigned int>(image[tX * cols + tY] * (gauss[i][j]));
    }
  return sum / 16;
}
