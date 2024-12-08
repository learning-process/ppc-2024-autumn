#include "mpi/vasilev_s_gaus3x3/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

std::vector<std::pair<int, int>> vasilev_s_gaus3x3_mpi::generateIndicesProcessedElements(int rows, int cols) {
  int total = (rows - 2) * (cols - 2);
  int row_start = 1;
  int row_end = rows - 1;
  int col_start = 1;
  int col_end = cols - 1;

  std::vector<std::pair<int, int>> indices;
  indices.reserve(total);

  for (int i = row_start; i < row_end; ++i) {
    for (int j = col_start; j < col_end; ++j) {
      indices.emplace_back(i, j);
    }
  }

  return indices;
}

void vasilev_s_gaus3x3_mpi::calculateIndicesSizesDispls(int total_elements, int cols, int num_proc,
                                                        std::vector<int>& sizes, std::vector<int>& displs) {
  sizes.resize(num_proc, 0);
  displs.resize(num_proc, 0);

  if (num_proc > total_elements) {
    for (int i = 0; i < total_elements; ++i) {
      sizes[i] = cols;
      displs[i] = i * cols;
    }
  } else {
    int a = total_elements / num_proc;
    int b = total_elements % num_proc;

    int offset = 0;
    for (int i = 0; i < num_proc; ++i) {
      if (b-- > 0) {
        sizes[i] = a + 1;
      } else {
        sizes[i] = a;
      }
      displs[i] = offset;
      offset += sizes[i];
    }
  }
}

std::vector<std::vector<std::pair<int, int>>> vasilev_s_gaus3x3_mpi::makeWorkersIndices(
    const std::vector<std::pair<int, int>>& indices, const std::vector<int>& sizes, const std::vector<int>& displs) {
  std::vector<std::vector<std::pair<int, int>>> result;
  result.reserve(sizes.size());

  size_t total_indices = indices.size();

  for (size_t i = 0; i < sizes.size(); ++i) {
    int size = sizes[i];
    int displ = displs[i];

    std::vector<std::pair<int, int>> group;
    group.reserve(size);

    int end = displ + size;
    if (end > static_cast<int>(total_indices)) {
      end = total_indices;
    }

    for (int index = displ; index < end; ++index) {
      group.push_back(indices[index]);
    }

    result.push_back(std::move(group));
  }

  return result;
}

void vasilev_s_gaus3x3_mpi::calculateMatrixSizesDispls(const std::vector<std::vector<std::pair<int, int>>>& matrix,
                                                       int cols, std::vector<int>& sizes, std::vector<int>& displs) {
  sizes.reserve(matrix.size());
  displs.reserve(matrix.size());

  for (const auto& vector_pairs : matrix) {
    if (vector_pairs.empty()) {
      displs.push_back(0);
      sizes.push_back(0);
    } else {
      int r1 = vector_pairs.front().first;
      int c1 = vector_pairs.front().second;
      int r2 = vector_pairs.back().first;
      int c2 = vector_pairs.back().second;

      int start_index = (r1 - 1) * cols + (c1 - 1);
      int end_index = (r2 + 1) * cols + (c2 + 1);

      int size = end_index - start_index + 1;

      displs.push_back(start_index);
      sizes.push_back(size);
    }
  }
}

std::vector<int> embedValuesWithZeros(const std::vector<int>& values, int rows, int cols) {
  int resultRows = rows - 2;
  int resultCols = cols - 2;
  std::vector<int> result(rows * cols, 0);

  for (int i = 0; i < resultRows; ++i) {
    int src_offset = i * resultCols;
    int dst_offset = (i + 1) * cols + 1;
    std::copy(values.begin() + src_offset, values.begin() + src_offset + resultCols, result.begin() + dst_offset);
  }
  return result;
}

bool vasilev_s_gaus3x3_mpi::Gaus3x3ParallelMPI::validation() {
  internal_order_test();
  if (world.rank() != 0) {
    return true;
  }

  if (!taskData || taskData->inputs.size() < 3 || taskData->inputs_count.size() < 3 || taskData->outputs.empty() ||
      taskData->outputs_count.empty()) {
    return false;
  }

  int num_rows = *reinterpret_cast<int*>(taskData->inputs[1]);
  int num_cols = *reinterpret_cast<int*>(taskData->inputs[2]);

  auto expected_matrix_size = static_cast<size_t>(num_rows * num_cols);

  return num_rows >= 3 && num_cols >= 3 && taskData->inputs_count[0] == expected_matrix_size &&
         taskData->outputs_count[0] == expected_matrix_size && taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool vasilev_s_gaus3x3_mpi::Gaus3x3ParallelMPI::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* matrix_data = reinterpret_cast<int*>(taskData->inputs[0]);
    int matrix_size = taskData->inputs_count[0];

    rows = *reinterpret_cast<int*>(taskData->inputs[1]);
    cols = *reinterpret_cast<int*>(taskData->inputs[2]);

    matrix.assign(matrix_data, matrix_data + matrix_size);

    result_vector.resize((rows - 2) * (cols - 2), 0);

    indices = vasilev_s_gaus3x3_mpi::generateIndicesProcessedElements(rows, cols);

    calculateIndicesSizesDispls((rows - 2) * (cols - 2), 1, world.size(), indices_sizes, indices_displs);

    auto worker_indicies = makeWorkersIndices(indices, indices_sizes, indices_displs);
    calculateMatrixSizesDispls(worker_indicies, cols, worker_sizes, worker_displs);
  }

  return true;
}

bool vasilev_s_gaus3x3_mpi::Gaus3x3ParallelMPI::run() {
  internal_order_test();

  boost::mpi::broadcast(world, cols, 0);
  boost::mpi::broadcast(world, worker_sizes, 0);
  boost::mpi::broadcast(world, worker_displs, 0);
  boost::mpi::broadcast(world, indices_sizes, 0);

  int local_vector_size = worker_sizes[world.rank()];
  int local_indices_size = indices_sizes[world.rank()];
  std::vector<int> local_vector(local_vector_size);
  std::vector<std::pair<int, int>> local_indecies(local_indices_size);

  if (world.rank() == 0) {
    boost::mpi::scatterv(world, matrix.data(), worker_sizes, worker_displs, local_vector.data(),
                         worker_sizes[world.rank()], 0);
    boost::mpi::scatterv(world, indices.data(), indices_sizes, indices_displs, local_indecies.data(),
                         indices_sizes[world.rank()], 0);
  } else {
    boost::mpi::scatterv(world, local_vector.data(), worker_sizes[world.rank()], 0);
    boost::mpi::scatterv(world, local_indecies.data(), indices_sizes[world.rank()], 0);
  }

  std::vector<int> local_result(indices_sizes[world.rank()]);
  for (int i = 0; i < indices_sizes[world.rank()]; i++) {
    auto [r, c] = local_indecies[i];
    int ind = (r - 1) * cols + (c - 1) - worker_displs[world.rank()];
    std::vector<int> process_vector(9);
    for (int j = 0; j < 3; j++) {
      process_vector[3 * j] = local_vector[ind + cols * j];
      process_vector[3 * j + 1] = local_vector[ind + cols * j + 1];
      process_vector[3 * j + 2] = local_vector[ind + cols * j + 2];
    }
    int res = applyGaussianKernel(process_vector);

    local_result[i] = res;
  }

  if (world.rank() == 0) {
    boost::mpi::gatherv(world, local_result.data(), local_result.size(), result_vector.data(), indices_sizes,
                        indices_displs, 0);
  } else {
    boost::mpi::gatherv(world, local_result.data(), local_result.size(), 0);
  }

  return true;
}

bool vasilev_s_gaus3x3_mpi::Gaus3x3ParallelMPI::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
    std::vector<int> result = embedValuesWithZeros(result_vector, rows, cols);
    std::copy(result.begin(), result.end(), output_data);
  }

  return true;
}

bool vasilev_s_gaus3x3_mpi::Gaus3x3SequentialMPI::validation() {
  internal_order_test();
  if (!taskData || taskData->inputs.size() < 3 || taskData->inputs_count.size() < 3 || taskData->outputs.empty() ||
      taskData->outputs_count.empty()) {
    return false;
  }

  int num_rows = *reinterpret_cast<int*>(taskData->inputs[1]);
  int num_cols = *reinterpret_cast<int*>(taskData->inputs[2]);

  auto expected_matrix_size = static_cast<size_t>(num_rows * num_cols);

  return num_rows >= 3 && num_cols >= 3 && taskData->inputs_count[0] == expected_matrix_size &&
         taskData->outputs_count[0] == expected_matrix_size && taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool vasilev_s_gaus3x3_mpi::Gaus3x3SequentialMPI::pre_processing() {
  internal_order_test();
  auto* matrix_data = reinterpret_cast<int*>(taskData->inputs[0]);
  int matrix_size = taskData->inputs_count[0];

  rows = *reinterpret_cast<int*>(taskData->inputs[1]);
  cols = *reinterpret_cast<int*>(taskData->inputs[2]);

  matrix.assign(matrix_data, matrix_data + matrix_size);

  int result_size = taskData->outputs_count[0];
  result_vector.resize(result_size, 0);

  return true;
}

bool vasilev_s_gaus3x3_mpi::Gaus3x3SequentialMPI::run() {
  internal_order_test();

  for (int row = 1; row < rows - 1; ++row) {
    for (int col = 1; col < cols - 1; ++col) {
      int baseIndex = row * cols + col;

      double result = 0;                                // Тут ошибка
      result += matrix[baseIndex - cols - 1] * 0.0625;  // Top-left
      result += matrix[baseIndex - cols] * 0.125;       // Top-center
      result += matrix[baseIndex - cols + 1] * 0.0625;  // Top-right
      result += matrix[baseIndex - 1] * 0.125;          // Middle-left
      result += matrix[baseIndex] * 0.25;               // Center
      result += matrix[baseIndex + 1] * 0.125;          // Middle-right
      result += matrix[baseIndex + cols - 1] * 0.0625;  // Bottom-left
      result += matrix[baseIndex + cols] * 0.125;       // Bottom-center
      result += matrix[baseIndex + cols + 1] * 0.0625;  // Bottom-right
      int finalResult = static_cast<int>(std::round(result));
      result_vector[baseIndex] = std::clamp(finalResult, 0, 255);
    }
  }

  return true;
}

bool vasilev_s_gaus3x3_mpi::Gaus3x3SequentialMPI::post_processing() {
  internal_order_test();

  auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(result_vector.begin(), result_vector.end(), output_data);

  return true;
}
