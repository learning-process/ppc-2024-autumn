#include "mpi/sedova_o_vertical_ribbon_scheme/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

void sedova_o_vertical_ribbon_scheme_mpi::calculate_distribution(int rows, int cols, int num_proc,
                                                                 std::vector<int>& sizes, std::vector<int>& displs) {
  sizes.resize(num_proc, 0);
  displs.resize(num_proc, -1);

  if (num_proc > cols) {
    for (int i = 0; i < cols; ++i) {
      sizes[i] = rows;
      displs[i] = i * rows;
    }
  } else {
    int a = cols / num_proc;
    int b = cols % num_proc;

    int offset = 0;
    for (int i = 0; i < num_proc; ++i) {
      if (b-- > 0) {
        sizes[i] = (a + 1) * rows;
      } else {
        sizes[i] = a * rows;
      }
      displs[i] = offset;
      offset += sizes[i];
    }
  }
}

bool sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI::validation() {
  internal_order_test();
  if (world.rank() != 0) return true;  // Only rank 0 performs validation
  bool valid_matrix = taskData->inputs[0] != nullptr && taskData->inputs_count[0] > 0;
  bool valid_vector = taskData->inputs[1] != nullptr && taskData->inputs_count[1] > 0;
  bool valid_dimensions = valid_matrix && valid_vector && taskData->inputs_count[0] % taskData->inputs_count[1] == 0;
  bool valid_result =
      valid_dimensions && taskData->outputs_count[0] == taskData->inputs_count[0] / taskData->inputs_count[1];
  return valid_result;
}

bool sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int* matrix_data = reinterpret_cast<int*>(taskData->inputs[0]);
    int matrix_size = taskData->inputs_count[0];
    int* vector_data = reinterpret_cast<int*>(taskData->inputs[1]);
    int vector_size = taskData->inputs_count[1];

    input_matrix_.assign(matrix_data, matrix_data + matrix_size);
    input_vector_.assign(vector_data, vector_data + vector_size);

    num_cols_ = input_vector_.size();
    num_rows_ = input_matrix_.size() / num_cols_;
    int result_size = taskData->outputs_count[0];
    result_vector_.resize(result_size, 0);

    calculate_distribution(num_cols_, num_rows_, world.size(), distribution, displacement);
  }
  return true;
}

bool sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI::run() {
  internal_order_test();

  mpi::broadcast(world, num_rows_, 0);      
  mpi::broadcast(world, input_vector_, 0); 
  mpi::broadcast(world, distribution, 0);

  int local_num_elements = distribution[world.rank()];
  int local_num_cols = local_num_elements / num_rows_;

  std::vector<int> local_matrix(local_num_elements);
  if (world.rank() == 0) {
    mpi::scatterv(world, input_matrix_.data(), distribution, displacement, local_matrix.data(), local_num_elements, 0);
  } else {
    mpi::scatterv(world, local_matrix.data(), local_num_elements, 0);
  }

  std::vector<int> local_result(local_num_cols, 0);

  for (int j = 0; j < local_num_cols; ++j) {  // j iterates over local columns
    for (int i = 0; i < num_rows_; ++i) {     // i iterates over local rows
      local_result[i] += local_matrix[i * local_num_cols + j] * input_vector_[j];
    }
  }

  std::vector<int> gather_counts;
  std::vector<int> gather_displacements;

  if (world.rank() == 0) {
    gather_counts.resize(world.size());
    gather_displacements.resize(world.size());

    for (int i = 0; i < world.size(); ++i) {
      int num_elements = distribution[i] / num_rows_;
      gather_counts[i] = num_elements;
    }

    gather_displacements[0] = 0;
    for (int i = 1; i < world.size(); ++i) {
      gather_displacements[i] = gather_displacements[i - 1] + gather_counts[i - 1];
    }
  }

  if (world.rank() == 0) {
    mpi::gatherv(world, local_result.data(), local_result.size(), result_vector_.data(), gather_counts, gather_displacements, 0);
  } else {
    mpi::gatherv(world, local_result.data(), local_result.size(), 0);
  }

  return true;
}
bool sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(result_vector_.begin(), result_vector_.end(), output_data);
  }
  return true;
}

bool sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI::validation() {
  internal_order_test();
  bool valid_matrix = taskData->inputs_count[0] > 0;
  bool valid_vector = taskData->inputs_count[1] > 0;
  bool valid_dimensions = valid_matrix && valid_vector && taskData->inputs_count[0] % taskData->inputs_count[1] == 0;
  bool valid_result =
      valid_dimensions && taskData->outputs_count[0] == taskData->inputs_count[0] / taskData->inputs_count[1];

  return valid_result;
}

bool sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI::pre_processing() {
  internal_order_test();

  int* matrix_data = reinterpret_cast<int*>(taskData->inputs[0]);
  int matrix_size = taskData->inputs_count[0];

  int* vector_data = reinterpret_cast<int*>(taskData->inputs[1]);
  int vector_size = taskData->inputs_count[1];

  input_matrix_.assign(matrix_data, matrix_data + matrix_size);
  input_vector_.assign(vector_data, vector_data + vector_size);

  num_cols_ = input_vector_.size();
  num_rows_ = input_matrix_.size() / num_cols_;

  int result_size = taskData->outputs_count[0];
  result_vector_.resize(result_size, 0);

  return true;
}

bool sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI::run() {
  internal_order_test();

  for (int j = 0; j < num_cols_; j++) {
    for (int i = 0; i < num_rows_; i++) {
      result_vector_[i] += input_matrix_[i * num_cols_ + j] * input_vector_[j];
    }
  }
  return true;
}

bool sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI::post_processing() {
  internal_order_test();

  int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(result_vector_.begin(), result_vector_.end(), output_data);

  return true;
}