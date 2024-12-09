#include "seq/chernykh_a_adjust_image_contrast/include/ops_seq.hpp"

#include <algorithm>

namespace chernykh_a_adjust_image_contrast_seq {

bool SequentialTask::validation() {
  internal_order_test();
  return contrast_factor >= 0.0 && contrast_factor <= 2.0 &&       // Contrast factor within [0.0, 2.0]
         taskData->inputs_count[0] > 0 &&                          // Input image must have at least 1 pixel
         taskData->inputs_count[0] == taskData->outputs_count[0];  // Input and output image sizes match
}

bool SequentialTask::pre_processing() {
  internal_order_test();
  auto* input_ptr = reinterpret_cast<Pixel*>(taskData->inputs[0]);
  auto input_size = taskData->inputs_count[0];
  auto result_size = taskData->outputs_count[0];
  input.assign(input_ptr, input_ptr + input_size);
  result.resize(result_size);
  return true;
}

bool SequentialTask::run() {
  internal_order_test();
  std::transform(input.begin(), input.end(), result.begin(),
                 [this](const Pixel& p) { return p.with_contrast(contrast_factor); });
  return true;
}

bool SequentialTask::post_processing() {
  internal_order_test();
  auto* result_ptr = reinterpret_cast<Pixel*>(taskData->outputs[0]);
  std::copy(result.begin(), result.end(), result_ptr);
  return true;
}

}  // namespace chernykh_a_adjust_image_contrast_seq
