
#include "mpi/vershinina_a_image_smoothing/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool vershinina_a_image_smoothing::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = reinterpret_cast<int*>(taskData->inputs[0]);
  rows = taskData->inputs_count[0];
  cols = taskData->inputs_count[1];
  output_.resize(rows * cols);
  return true;
}

bool vershinina_a_image_smoothing::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs.size() == 1 && taskData->inputs_count.size() == 2 && taskData->outputs.size() == 1;
}

bool vershinina_a_image_smoothing::TestMPITaskSequential::run() {
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

bool vershinina_a_image_smoothing::TestMPITaskSequential::post_processing() {
  internal_order_test();
  std::copy(output_.begin(), output_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  return true;
}

bool vershinina_a_image_smoothing::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* pr = reinterpret_cast<int*>(taskData->inputs[0]);
    input_.resize(taskData->inputs_count[0] * taskData->inputs_count[1]);
    std::copy(pr, pr + input_.size(), input_.begin());
  }
  return true;
}

bool vershinina_a_image_smoothing::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs.size() == 1 && taskData->inputs_count.size() == 2 && taskData->outputs.size() == 1;
  }
  return true;
}

bool vershinina_a_image_smoothing::TestMPITaskParallel::run() {
  internal_order_test();
  if (world.rank() == 0) {
    rows = taskData->inputs_count[0];
    cols = taskData->inputs_count[1];
    rows_per_proc = rows / std::min(world.size(), rows);
    remainder = rows % world.size();
  }

  broadcast(world, rows, 0);
  broadcast(world, cols, 0);
  broadcast(world, local_rows, 0);
  broadcast(world, rows_per_proc, 0);
  broadcast(world, remainder, 0);

  local_input_sizes.resize(world.size(), 0);
  std::vector<int> send(world.size(), 0);

  if (world.rank() == 0) {
    if (world.size() > 1) {
      if (rows >= world.size()) {
        for (int i = 0; i < std::min(world.size(), rows); i++) {
          if (i == world.size() - 1) {
            local_input_sizes[i] = (rows_per_proc + 1 + remainder) * cols;
          } else if (i == 0) {
            local_input_sizes[i] = (rows_per_proc + 1) * cols;
          } else {
            local_input_sizes[i] = (rows_per_proc + 2) * cols;
          }
        }
      } else {
        for (int i = 0; i < std::min(world.size(), rows); i++) {
          if ((i == 0) || (i == rows - 1)) {
            local_input_sizes[i] = (rows_per_proc + 1) * cols;
          } else {
            local_input_sizes[i] = (rows_per_proc + 2) * cols;
          }
        }
      }
    } else {
      local_input_sizes[0] = rows * cols;
    }
    if (world.size() > 1) {
      for (int i = 1; i < std::min(world.size(), rows); i++) {
        send[i] = i * rows_per_proc * cols - 1 * cols;
      }
    }
  }

  broadcast(world, local_input_sizes.data(), local_input_sizes.size(), 0);
  local_input_.resize(local_input_sizes[world.rank()]);
  boost::mpi::scatterv(world, input_.data(), local_input_sizes, send, local_input_.data(),
                       local_input_sizes[world.rank()], 0);
  local_rows = local_input_sizes[world.rank()] / cols;

  if (world.rank() == 0) {
    output_.resize(rows * cols);
  }

  int offset_top = (world.rank() == 0) ? 0 : 1;
  int offset_bottom = (world.rank() == std::min(world.size(), rows) - 1) ? 0 : 1;

  if (world.size() == 1) {
    offset_top = 0;
    offset_bottom = 0;
  }

  if (local_input_sizes[world.rank()] != 0) {
    local_output_.resize((local_rows - offset_top - offset_bottom) * cols);
  }

  for (int i = offset_top; i < local_rows - offset_bottom; ++i) {
    for (int j = 0; j < cols; ++j) {
      int sum = 0;
      int c = 0;
      for (int row = std::max(0, i - 1); row <= std::min(i + 1, local_rows - 1); row++) {
        for (int col = std::max(0, j - 1); col <= std::min(j + 1, cols - 1); col++) {
          sum += local_input_[row * cols + col];
          c++;
        }
      }
      local_output_[(i - offset_top) * cols + j] = sum / c;
    }
  }

  int* output_data = output_.data();
  int* local_output_data = local_output_.data();

  for (int i = 0; i < std::min(world.size(), rows); i++) {
    if (i == world.size() - 1) {
      local_input_sizes[i] = (rows_per_proc + remainder) * cols;
    } else {
      local_input_sizes[i] = rows_per_proc * cols;
    }
  }

  boost::mpi::gatherv(world, local_output_data, local_output_.size(), output_data, local_input_sizes, 0);
  return true;
}

bool vershinina_a_image_smoothing::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    std::copy(output_.begin(), output_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }
  return true;
}
