#include "seq/milovankin_m_component_labeling/include/component_labeling_seq.hpp"

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace milovankin_m_component_labeling_seq {

bool ComponentLabelingSeq::validation() {
  internal_order_test();

  return !taskData->inputs.empty() && !taskData->outputs.empty() && taskData->inputs_count.size() == 2 &&
         taskData->outputs_count.size() == 2 && taskData->inputs_count[0] == taskData->outputs_count[0] &&
         taskData->inputs_count[1] == taskData->outputs_count[1];
}

bool ComponentLabelingSeq::pre_processing() {
  internal_order_test();

  rows = taskData->inputs_count[0];
  cols = taskData->inputs_count[1];

  std::size_t total_pixels = rows * cols;
  input_image_.resize(total_pixels);
  std::copy_n(reinterpret_cast<uint8_t*>(taskData->inputs[0]), total_pixels, input_image_.begin());

  return true;
}

bool ComponentLabelingSeq::run() {
  internal_order_test();

  std::unordered_map<uint32_t, uint32_t> label_equivalences;
  uint32_t next_label = 1;

  labels_.resize(rows * cols, 0);

  auto linear_index = [&cols = this->cols](std::size_t row, std::size_t col) -> std::size_t {
    return row * cols + col;
  };

  // Assign labels and record equivalences
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      if (input_image_[linear_index(row, col)] == 0) {
        continue;
      }

      uint32_t label_B = (col > 0) ? labels_[linear_index(row, col - 1)] : 0;
      uint32_t label_C = (row > 0) ? labels_[linear_index(row - 1, col)] : 0;
      uint32_t label_D = (row > 0 && col > 0) ? labels_[linear_index(row - 1, col - 1)] : 0;

      if (label_B == 0 && label_C == 0 && label_D == 0) {
        labels_[linear_index(row, col)] = next_label++;
      } else {
        uint32_t min_label = std::min({label_B, label_C, label_D}, [](uint32_t a, uint32_t b) {
          if (a == 0) {
            return false;
          }
          if (b == 0) {
            return true;
          }
          return a < b;  // min higher than 0
        });

        labels_[linear_index(row, col)] = min_label;

        for (uint32_t lbl : {label_B, label_C, label_D}) {
          if (lbl != 0 && lbl != min_label) {
            label_equivalences[std::max(lbl, min_label)] = std::min(lbl, min_label);
          }
        }
      }
    }
  }

  // Resolve label equivalences
  for (auto& label : labels_) {
    while (label_equivalences.contains(label)) {
      label = label_equivalences[label];
    }
  }

  return true;
}

bool ComponentLabelingSeq::post_processing() {
  internal_order_test();

  std::size_t total_pixels = rows * cols;
  std::copy_n(labels_.data(), total_pixels, reinterpret_cast<uint32_t*>(taskData->outputs[0]));

  return true;
}

}  // namespace milovankin_m_component_labeling_seq