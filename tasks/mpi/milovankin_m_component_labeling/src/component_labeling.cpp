#include "mpi/milovankin_m_component_labeling/include/component_labeling.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

namespace milovankin_m_component_labeling_mpi {

// ----------------------------------------------------------------
//                      Sequential version
// ----------------------------------------------------------------

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

// ----------------------------------------------------------------
//                      Parallel version
// ----------------------------------------------------------------

bool ComponentLabelingPar::validation() {
  internal_order_test();
  return !taskData->inputs.empty() && !taskData->outputs.empty() && taskData->inputs_count.size() == 2 &&
         taskData->outputs_count.size() == 2 && taskData->inputs_count[0] == taskData->outputs_count[0] &&
         taskData->inputs_count[1] == taskData->outputs_count[1];
}

bool ComponentLabelingPar::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    rows = taskData->inputs_count[0];
    cols = taskData->inputs_count[1];
    int total_size = rows * cols;
    input_image_.resize(total_size);

    std::copy_n(reinterpret_cast<uint8_t*>(taskData->inputs[0]), total_size, input_image_.begin());
  }

  return true;
}

bool ComponentLabelingPar::run() {
  internal_order_test();

  boost::mpi::broadcast(world, rows, 0);
  boost::mpi::broadcast(world, cols, 0);

  int rank = world.rank();
  int world_size = world.size();

  std::vector<int> send_counts(world_size, 0);
  std::vector<int> offsets(world_size, 0);

  // Each segment starts at the beginning of a row
  int row_offset = 0;
  for (int i = 0; i < world_size; ++i) {
    int rows_for_proc = rows / world_size;
    send_counts[i] = rows_for_proc * cols;
    offsets[i] = row_offset * cols;
    row_offset += rows_for_proc;

    if (i == world_size - 1) send_counts[i] += (cols * rows - (rows / world_size) * cols * world_size);
  }

  // Determine the local size for the current process
  int local_size = send_counts[rank];
  local_image_.resize(local_size);

  // Scatter the data
  boost::mpi::scatterv(world, input_image_.data(), send_counts, offsets, local_image_.data(), local_size, 0);

  // auto linear_index = [this](std::size_t row, std::size_t col) -> std::size_t { return row * cols + col; };
  auto coord_index = [this](std::size_t index) -> std::pair<std::size_t, std::size_t> {
    return std::make_pair(index / cols, index % cols);
  };  // row; column

  //
  // Run algorithm for each segment
  //

  uint32_t next_label = 1 + offsets[rank];
  std::vector<uint32_t> local_labels_(local_size);

  // Assign labels and record equivalences
  std::unordered_map<uint32_t, uint32_t> local_label_equivalences;
  for (std::size_t i = 0; i < (size_t)local_size; ++i) {
    if (local_image_[i] == 0) continue;

    std::pair<std::size_t, std::size_t> coord_r_c = coord_index(i);
    size_t row = coord_r_c.first;
    size_t col = coord_r_c.second;

    // D C
    // B A
    uint32_t label_B = (col > 0) ? local_labels_[i - 1] : 0;
    uint32_t label_C = (row > 0) ? local_labels_[i - cols] : 0;
    uint32_t label_D = (row > 0 && col > 0) ? local_labels_[i - cols - 1] : 0;

    if (label_B == 0 && label_C == 0 && label_D == 0) {
      local_labels_[i] = next_label++;
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
      local_labels_[i] = min_label;

      for (uint32_t lbl : {label_B, label_C, label_D}) {
        if (lbl != 0 && lbl != min_label) {
          local_label_equivalences[std::max(lbl, min_label)] = std::min(lbl, min_label);
        }
      }
    }
  }
  // Update table according to equivalences
  for (auto& label : local_labels_) {
    while (local_label_equivalences.contains(label)) {
      label = local_label_equivalences[label];
    }
  }

  if (rank == 0) labels_.resize(rows * cols);
  boost::mpi::gatherv(world, local_labels_, labels_.data(), send_counts, offsets, 0);

  // Merge labels from nearby sections
  if (rank == 0) {
    std::unordered_map<uint32_t, uint32_t> label_equivalences;

    for (int section = 1; section < world_size; ++section) {
      int border = offsets[section];

      for (int lbl = border; (size_t)lbl < border + cols; ++lbl) {
        if (labels_[lbl] == 0) continue;

        std::pair<std::size_t, std::size_t> coord_r_c = coord_index(lbl);
        size_t row = coord_r_c.first;
        size_t col = coord_r_c.second;

        // D C
        // B A
        uint32_t label_B = (col > 0) ? labels_[lbl - 1] : 0;
        uint32_t label_C = (row > 0) ? labels_[lbl - cols] : 0;
        uint32_t label_D = (row > 0 && col > 0) ? labels_[lbl - cols - 1] : 0;

        if (label_B != 0 || label_C != 0 || label_D != 0) {
          uint32_t min_label = std::min({label_B, label_C, label_D}, [](uint32_t a, uint32_t b) {
            if (a == 0) {
              return false;
            }
            if (b == 0) {
              return true;
            }
            return a < b;  // min higher than 0
          });
          labels_[lbl] = min_label;

          for (uint32_t lbl_ : {label_B, label_C, label_D}) {
            if (lbl_ != 0 && lbl_ != min_label) {
              label_equivalences[std::max(lbl_, min_label)] = std::min(lbl_, min_label);
            }
          }
        }
      }
    }
    // Update table according to equivalences
    for (auto& label : labels_) {
      while (label_equivalences.contains(label)) {
        label = label_equivalences[label];
      }
    }
  }

  return true;
}

bool ComponentLabelingPar::post_processing() {
  internal_order_test();

  if (world.rank() == 0) std::copy_n(labels_.data(), rows * cols, reinterpret_cast<uint32_t*>(taskData->outputs[0]));
  return true;
}

}  // namespace milovankin_m_component_labeling_mpi
