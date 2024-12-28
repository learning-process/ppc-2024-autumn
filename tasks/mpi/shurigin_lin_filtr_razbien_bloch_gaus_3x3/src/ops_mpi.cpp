#include "mpi/shurigin_lin_filtr_razbien_bloch_gaus_3x3/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <numeric>
#include <vector>

namespace shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi {

std::vector<std::pair<int, int>> computeProcessingIndices(int rows, int cols) {
  std::vector<std::pair<int, int>> indices;
  indices.reserve((rows - 2) * (cols - 2));

  int top = 1;
  int bottom = rows - 2;
  int left = 1;
  int right = cols - 2;

  while (top <= bottom && left <= right) {
    // Top row
    for (int j = left; j <= right; j++) {
      indices.emplace_back(top, j);
    }
    top++;

    // Right column
    for (int i = top; i <= bottom; i++) {
      indices.emplace_back(i, right);
    }
    right--;

    if (top <= bottom) {
      // Bottom row
      for (int j = right; j >= left; j--) {
        indices.emplace_back(bottom, j);
      }
      bottom--;
    }

    if (left <= right) {
      // Left column
      for (int i = bottom; i >= top; i--) {
        indices.emplace_back(i, left);
      }
      left++;
    }
  }

  std::sort(indices.begin(), indices.end());
  return indices;
}

void calculateDistribution(int total_elements, int cols, int num_proc, std::vector<int>& block_sizes,
                           std::vector<int>& block_offsets) {
  block_sizes.clear();
  block_offsets.clear();
  block_sizes.resize(num_proc);
  block_offsets.resize(num_proc);

  int base_block_size = total_elements / num_proc;
  int remaining = total_elements % num_proc;

  int current_offset = 0;

  for (int i = 0; i < num_proc; ++i) {
    block_sizes[i] = base_block_size + (i < remaining ? 1 : 0);
    block_offsets[i] = current_offset;
    current_offset += block_sizes[i];  // del exp
  }
}

std::vector<std::vector<std::pair<int, int>>> distributeWorkload(const std::vector<std::pair<int, int>>& indices,
                                                                 const std::vector<int>& sizes,
                                                                 const std::vector<int>& offsets) {
  std::vector<std::vector<std::pair<int, int>>> distributed_work;
  distributed_work.reserve(sizes.size());

  size_t total_indices = indices.size();

  for (size_t i = 0; i < sizes.size(); ++i) {
    std::vector<std::pair<int, int>> worker_chunk;
    worker_chunk.reserve(sizes[i]);

    int end_pos = offsets[i] + sizes[i];
    if (end_pos > static_cast<int>(total_indices)) {
      end_pos = total_indices;
    }

    for (int idx = offsets[i]; idx < end_pos; ++idx) {
      worker_chunk.push_back(indices[idx]);
    }

    distributed_work.push_back(std::move(worker_chunk));
  }

  return distributed_work;
}

void computeBlockDistribution(const std::vector<std::vector<std::pair<int, int>>>& data, int cols,
                              std::vector<int>& sizes, std::vector<int>& offsets) {
  sizes.reserve(data.size());
  offsets.reserve(data.size());

  for (const auto& chunk : data) {
    if (chunk.empty()) {
      offsets.push_back(0);
      sizes.push_back(0);
      continue;
    }

    auto first_elem = chunk.front();
    auto last_elem = chunk.back();

    int start_idx = (first_elem.first - 1) * cols + (first_elem.second - 1);
    int end_idx = (last_elem.first + 1) * cols + (last_elem.second + 1);

    offsets.push_back(start_idx);
    sizes.push_back(end_idx - start_idx + 1);
  }
}

std::vector<int> padMatrixWithZeros(const std::vector<int>& core_values, int rows, int cols) {
  int inner_rows = rows - 2;
  int inner_cols = cols - 2;

  std::vector<int> padded_matrix(rows * cols, 0);

  for (int i = 0; i < inner_rows; ++i) {
    int src_pos = i * inner_cols;
    int dst_pos = (i + 1) * cols + 1;
    std::copy(core_values.begin() + src_pos, core_values.begin() + src_pos + inner_cols,
              padded_matrix.begin() + dst_pos);
  }
  return padded_matrix;
}

namespace {
const double KERNEL[3][3] = {{0.0625, 0.125, 0.0625}, {0.125, 0.25, 0.125}, {0.0625, 0.125, 0.0625}};

int applyFilter(const std::vector<int>& input, int rows, int cols, int x, int y) {
  double result = 0.0;
  for (int i = -1; i <= 1; ++i) {
    for (int j = -1; j <= 1; ++j) {
      int pos = (x + i) * cols + (y + j);
      result += input[pos] * KERNEL[i + 1][j + 1];
    }
  }
  return static_cast<int>(result);
}
}  // namespace

bool TaskMpi::validation() {
  internal_order_test();
  if (comm.rank() != 0) {
    return true;
  }

  if (!taskData || taskData->inputs.size() < 3 || taskData->inputs_count.size() < 3 || taskData->outputs.empty() ||
      taskData->outputs_count.empty()) {
    return false;
  }

  int num_rows = *reinterpret_cast<int*>(taskData->inputs[1]);
  int num_cols = *reinterpret_cast<int*>(taskData->inputs[2]);
  auto expected_size = static_cast<size_t>(num_rows * num_cols);

  return num_rows >= 3 && num_cols >= 3 && taskData->inputs_count[0] == expected_size &&
         taskData->outputs_count[0] == expected_size && taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool TaskMpi::pre_processing() {
  internal_order_test();

  if (comm.rank() == 0) {
    auto* data_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    int data_size = taskData->inputs_count[0];

    height = *reinterpret_cast<int*>(taskData->inputs[1]);
    width = *reinterpret_cast<int*>(taskData->inputs[2]);

    input_data.assign(data_ptr, data_ptr + data_size);
    processed_data.resize((height - 2) * (width - 2), 0);

    processing_indices = computeProcessingIndices(height, width);
    calculateDistribution((height - 2) * (width - 2), 1, comm.size(), block_sizes, block_offsets);

    auto distributed_indices = distributeWorkload(processing_indices, block_sizes, block_offsets);
    computeBlockDistribution(distributed_indices, width, data_sizes, data_offsets);
  }

  return true;
}

bool TaskMpi::run() {
  internal_order_test();

  boost::mpi::broadcast(comm, width, 0);
  boost::mpi::broadcast(comm, data_sizes, 0);
  boost::mpi::broadcast(comm, data_offsets, 0);
  boost::mpi::broadcast(comm, block_sizes, 0);

  int local_data_size = data_sizes[comm.rank()];
  int local_block_size = block_sizes[comm.rank()];
  std::vector<int> local_data(local_data_size);
  std::vector<std::pair<int, int>> local_coords(local_block_size);

  if (comm.rank() == 0) {
    boost::mpi::scatterv(comm, input_data.data(), data_sizes, data_offsets, local_data.data(), data_sizes[comm.rank()],
                         0);
    boost::mpi::scatterv(comm, processing_indices.data(), block_sizes, block_offsets, local_coords.data(),
                         block_sizes[comm.rank()], 0);
  } else {
    boost::mpi::scatterv(comm, local_data.data(), data_sizes[comm.rank()], 0);
    boost::mpi::scatterv(comm, local_coords.data(), block_sizes[comm.rank()], 0);
  }

  std::vector<int> local_result(block_sizes[comm.rank()]);
  for (int i = 0; i < block_sizes[comm.rank()]; i++) {
    auto [row, col] = local_coords[i];
    int base_idx = (row - 1) * width + (col - 1) - data_offsets[comm.rank()];

    std::vector<int> neighborhood(9);
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        neighborhood[j * 3 + k] = local_data[base_idx + width * j + k];
      }
    }

    local_result[i] = applyFilter(neighborhood, 3, 3, 1, 1);
  }

  if (comm.rank() == 0) {
    boost::mpi::gatherv(comm, local_result.data(), local_result.size(), processed_data.data(), block_sizes,
                        block_offsets, 0);
  } else {
    boost::mpi::gatherv(comm, local_result.data(), local_result.size(), 0);
  }

  return true;
}

bool TaskMpi::post_processing() {
  internal_order_test();

  if (comm.rank() == 0) {
    auto* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
    std::vector<int> final_result = padMatrixWithZeros(processed_data, height, width);
    std::copy(final_result.begin(), final_result.end(), output_ptr);
  }

  return true;
}

bool TaskSeq::validation() {
  internal_order_test();
  if (!taskData || taskData->inputs.size() < 3 || taskData->inputs_count.size() < 3 || taskData->outputs.empty() ||
      taskData->outputs_count.empty()) {
    return false;
  }

  num_rows = *reinterpret_cast<int*>(taskData->inputs[1]);
  num_cols = *reinterpret_cast<int*>(taskData->inputs[2]);

  if (num_rows < 3 || num_cols < 3) {
    return false;
  }

  auto expected_size = static_cast<size_t>(num_rows * num_cols);
  return taskData->inputs_count[0] == expected_size && taskData->outputs_count[0] == expected_size &&
         taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool TaskSeq::pre_processing() {
  internal_order_test();

  const auto* source_pixels = reinterpret_cast<const int*>(taskData->inputs[0]);
  const size_t total_pixels = taskData->inputs_count[0];

  input_data = std::vector<int>(source_pixels, source_pixels + total_pixels);
  output_data.resize(total_pixels);

  return true;
}

bool TaskSeq::run() {
  internal_order_test();

  for (int r = 1; r < num_rows - 1; ++r) {
    for (int c = 1; c < num_cols - 1; ++c) {
      std::vector<int> neighborhood(9);
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          neighborhood[i * 3 + j] = input_data[(r + i - 1) * num_cols + (c + j - 1)];
        }
      }
      output_data[r * num_cols + c] = applyFilter(neighborhood, 3, 3, 1, 1);
    }
  }

  return true;
}

bool TaskSeq::post_processing() {
  internal_order_test();

  auto* result_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(output_data.begin(), output_data.end(), result_data);

  return true;
}

}  // namespace shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi