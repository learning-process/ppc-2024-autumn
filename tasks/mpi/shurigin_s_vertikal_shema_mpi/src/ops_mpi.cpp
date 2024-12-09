#include "mpi/shurigin_s_vertikal_shema_mpi/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

namespace shurigin_s_vertikal_shema {

void calculate_distribution(int rows, int cols, int num_proc, std::vector<int>& sizes, std::vector<int>& displs) {
  sizes.resize(num_proc, 0);
  displs.resize(num_proc, -1);

  if (num_proc > cols) {
    for (int i = 0; i < cols; ++i) {
      sizes[i] = rows;
      displs[i] = i * rows;
    }
  } else {
    int cols_per_proc = cols / num_proc;
    int extra_cols = cols % num_proc;

    int offset = 0;
    for (int i = 0; i < num_proc; ++i) {
      if (extra_cols > 0) {
        sizes[i] = (cols_per_proc + 1) * rows;
        --extra_cols;
      } else {
        sizes[i] = cols_per_proc * rows;
      }
      displs[i] = offset;
      offset += sizes[i];
    }
  }
}

bool TestTaskMPI::validation() {
  if (world.rank() != 0) return true;

  if (!taskData) {
    return false;
  }

  bool valid_matrix = taskData->inputs[0] != nullptr && taskData->inputs_count[0] > 0;
  bool valid_vector = taskData->inputs[1] != nullptr && taskData->inputs_count[1] > 0;

  if (!valid_matrix || !valid_vector) {
    return false;
  }

  const size_t matrix_size = taskData->inputs_count[0];
  const size_t vector_size = taskData->inputs_count[1];

  if (matrix_size == 0 || vector_size == 0) {
    return false;
  }

  if (matrix_size % vector_size != 0) {
    return false;
  }

  const size_t num_rows = matrix_size / vector_size;
  const size_t num_procs = world.size();

  if (num_procs > vector_size) {
    if (static_cast<size_t>(world.rank()) >= vector_size) {
      return true;
    }
  }

  return taskData->outputs_count[0] == num_rows;
}

bool TestTaskMPI::pre_processing() {
  if (world.rank() == 0) {
    if (!taskData || taskData->inputs[0] == nullptr || taskData->inputs[1] == nullptr ||
        taskData->outputs[0] == nullptr) {
      return false;
    }

    int* matrix_data = reinterpret_cast<int*>(taskData->inputs[0]);
    int matrix_size = taskData->inputs_count[0];
    int* vector_data = reinterpret_cast<int*>(taskData->inputs[1]);
    int vector_size = taskData->inputs_count[1];

    num_cols_ = vector_size;
    num_rows_ = matrix_size / num_cols_;

    input_matrix_.assign(matrix_data, matrix_data + matrix_size);
    input_vector_.assign(vector_data, vector_data + vector_size);
    result_vector_.resize(num_rows_, 0);

    calculate_distribution(num_rows_, num_cols_, world.size(), distribution, displacement);
  }

  boost::mpi::broadcast(world, num_rows_, 0);
  boost::mpi::broadcast(world, num_cols_, 0);
  boost::mpi::broadcast(world, distribution, 0);
  boost::mpi::broadcast(world, displacement, 0);

  if (world.rank() != 0) {
    input_matrix_.resize(num_rows_ * num_cols_);
    input_vector_.resize(num_cols_);
    result_vector_.resize(num_rows_);
  }

  return true;
}

bool TestTaskMPI::run() {
  boost::mpi::broadcast(world, input_matrix_, 0);
  boost::mpi::broadcast(world, input_vector_, 0);

  int local_start_col = displacement[world.rank()] / num_rows_;
  int local_cols = distribution[world.rank()] / num_rows_;
  std::vector<int> local_result(num_rows_, 0);

  for (int i = 0; i < num_rows_; ++i) {
    for (int j = 0; j < local_cols; ++j) {
      int global_col = local_start_col + j;
      int matrix_val = input_matrix_[i * num_cols_ + global_col];
      int vector_val = input_vector_[global_col];
      local_result[i] += matrix_val * vector_val;
    }
  }

  if (world.rank() == 0) {
    result_vector_.assign(num_rows_, 0);
  }

  boost::mpi::reduce(world, local_result.data(), num_rows_, result_vector_.data(), std::plus<>(), 0);

  return true;
}

bool TestTaskMPI::post_processing() {
  if (world.rank() == 0) {
    int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(result_vector_.begin(), result_vector_.end(), output_data);
  }
  return true;
}

bool TestTaskSequential::validation() {
  return taskData && taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0;
}

bool TestTaskSequential::pre_processing() {
  matrix_data_ = reinterpret_cast<int*>(taskData->inputs[0]);
  int matrix_size = taskData->inputs_count[0];

  int* vector_data = reinterpret_cast<int*>(taskData->inputs[1]);
  num_cols_ = taskData->inputs_count[1];

  num_rows_ = matrix_size / num_cols_;

  input_vector_.assign(vector_data, vector_data + num_cols_);

  result_vector_.assign(num_rows_, 0);

  return true;
}

bool TestTaskSequential::run() {
  for (int i = 0; i < num_rows_; ++i) {
    result_vector_[i] = 0;
    for (int j = 0; j < num_cols_; ++j) {
      result_vector_[i] += matrix_data_[i * num_cols_ + j] * input_vector_[j];
    }
  }

  return true;
}

bool TestTaskSequential::post_processing() {
  int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(result_vector_.begin(), result_vector_.end(), output_data);

  return true;
}

}  // namespace shurigin_s_vertikal_shema
