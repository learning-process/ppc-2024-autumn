#include "mpi/milovankin_m_component_labeling/include/component_labeling.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>

namespace milovankin_m_component_labeling_mpi {

// ----------------------------------------------------------------
//                      Sequential version
// ----------------------------------------------------------------

bool ComponentLabelingSeq::validation() {
  internal_order_test();
  return !taskData->inputs.empty() && !taskData->outputs.empty();
}

bool ComponentLabelingSeq::pre_processing() {
  internal_order_test();

  rows = taskData->inputs_count[0];
  cols = taskData->inputs_count[1];

  std::size_t total_pixels = rows * cols;
  input_image_.resize(total_pixels);
  std::copy_n(reinterpret_cast<uint8_t*>(taskData->inputs[0]), total_pixels, input_image_.begin());
  labels_.resize(total_pixels, 0);

  return true;
}

bool ComponentLabelingSeq::run() {
  internal_order_test();

  uint32_t label = 1;

  auto linear_index = [this](std::size_t row, std::size_t col) -> std::size_t { return row * cols + col; };

  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      if (input_image_[linear_index(row, col)] == 0) {
        continue;
      }

      uint32_t max_label = 0;

      // Check 4 neighbors
      for (int dr : {-1, 0, 1}) {
        for (int dc : {-1, 0, 1}) {
          if ((dr == 0 && dc == 0) || (dr != 0 && dc != 0)) continue;
          std::size_t n_row = row + dr;
          std::size_t n_col = col + dc;
          if (n_row < rows && n_col < cols) {
            max_label = std::max(max_label, labels_[linear_index(n_row, n_col)]);
          }
        }
      }

      // Bottom left neighbor
      if (row > 0 && col > 0) {
        max_label = std::max(max_label, labels_[linear_index(row - 1, col - 1)]);
      }

      // Top right neighbor
      if (row + 1 < rows && col + 1 < cols) {
        max_label = std::max(max_label, labels_[linear_index(row + 1, col + 1)]);
      }

      if (max_label == 0) {
        labels_[linear_index(row, col)] = label++;
      } else {
        labels_[linear_index(row, col)] = max_label;
      }
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

// ----------------------------------------------------------------
//                      Parallel version
// ----------------------------------------------------------------

bool ComponentLabelingPar::validation() {
  internal_order_test();
  return !taskData->inputs.empty() && !taskData->outputs.empty();
}

bool ComponentLabelingPar::pre_processing() {
  internal_order_test();

  rows = taskData->inputs_count[0];
  cols = taskData->inputs_count[1];

  std::size_t total_pixels = rows * cols;
  input_image_.resize(total_pixels);
  std::copy_n(reinterpret_cast<uint8_t*>(taskData->inputs[0]), total_pixels, input_image_.begin());
  labels_.resize(total_pixels, 0);

  return true;
}

bool ComponentLabelingPar::run() {
  internal_order_test();

  uint32_t label = 1;

  size_t rows_per_proc = rows / world.size();
  size_t start_row = rows_per_proc * world.rank();
  size_t end_row = (world.rank() == world.size() - 1) ? rows : start_row + rows_per_proc;

  auto linear_index = [this](std::size_t row, std::size_t col) -> std::size_t { return row * cols + col; };

  // Local computation of labels for the assigned rows
  for (std::size_t row = start_row; row < end_row; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      if (input_image_[linear_index(row, col)] == 0) {
        continue;
      }

      uint32_t max_label = 0;

      for (int dr : {-1, 0, 1}) {
        for (int dc : {-1, 0, 1}) {
          if ((dr == 0 && dc == 0) || (dr != 0 && dc != 0)) continue;
          std::size_t n_row = row + dr;
          std::size_t n_col = col + dc;
          if (n_row >= start_row && n_row < end_row && n_col < cols) {
            max_label = std::max(max_label, labels_[linear_index(n_row, n_col)]);
          }
        }
      }

      if (row > 0 && col > 0) {
        max_label = std::max(max_label, labels_[linear_index(row - 1, col - 1)]);
      }

      if (row + 1 < rows && col + 1 < cols) {
        max_label = std::max(max_label, labels_[linear_index(row + 1, col + 1)]);
      }

      if (max_label == 0) {
        labels_[linear_index(row, col)] = label++;
      } else {
        labels_[linear_index(row, col)] = max_label;
      }
    }
  }

  boost::mpi::all_gather(world, labels_.data() + start_row * cols, (end_row - start_row) * cols, labels_.data());

  return true;
}

bool ComponentLabelingPar::post_processing() {
  internal_order_test();

  std::copy_n(labels_.data(), rows * cols, reinterpret_cast<uint32_t*>(taskData->outputs[0]));
  return true;
}

}  // namespace milovankin_m_component_labeling_mpi
