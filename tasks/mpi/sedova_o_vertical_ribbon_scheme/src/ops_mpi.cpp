#include "mpi/sedova_o_vertical_ribbon_scheme/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>
#include <random>
#include <cassert>

bool sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] > 1 && taskData->inputs_count[1] > 0;
  }
  return true;
}
bool sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    if (!taskData || taskData->inputs[0] == nullptr || taskData->inputs[1] == nullptr ||
        taskData->outputs[0] == nullptr) {
      return false;
    }

    int* input_matrix_ = reinterpret_cast<int*>(taskData->inputs[0]);
    int* input_vector_ = reinterpret_cast<int*>(taskData->inputs[1]);

    int count = taskData->inputs_count[0];
    cols_ = taskData->inputs_count[1];
    rows_ = count / cols_;

    input_matrix_1.assign(input_matrix_, input_matrix_ + count);
    input_vector_1.assign(input_vector_, input_vector_ + cols_);
    result_vector_.resize(rows_, 0);

    distribution.resize(world.size(), 0);
    displacement.resize(world.size(), -1);

    if (world.size() > cols_) {
      for (int i = 0; i < cols_; ++i) {
        distribution[i] = rows_;
        displacement[i] = i * rows_;
      }
    } else {
      int cols_per_proc = cols_ / world.size();
      int ost = cols_ % world.size();

      int offset = 0;
      for (int i = 0; i < world.size(); ++i) {
        if (ost > 0) {
          distribution[i] = (cols_per_proc + 1) * rows_;
          --ost;
        } else {
          distribution[i] = cols_per_proc * rows_;
        }
        displacement[i] = offset;
        offset += distribution[i];
      }
    }
  }

  if (world.rank() != 0) {
    input_matrix_1.resize(rows_ * cols_, 0);
    input_vector_1.resize(cols_, 0);
    result_vector_.resize(rows_, 0);
  }

  return true;
}

bool sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI::run() {
  internal_order_test();

  boost::mpi::broadcast(world, cols_, 0);
  boost::mpi::broadcast(world, rows_, 0);
  boost::mpi::broadcast(world, distribution, 0);
  boost::mpi::broadcast(world, displacement, 0);
  boost::mpi::broadcast(world, input_matrix_1, 0);
  boost::mpi::broadcast(world, input_vector_1, 0);

  int local_start_cols_ = displacement[world.rank()] / rows_;
  int local_cols_ = distribution[world.rank()] / rows_;
  std::vector<int> local_result_vector_(rows_, 0);

  for (int i = 0; i < local_cols_; ++i) {
    for (int j = 0; j < rows_; ++j) {
      int global_cols_ = local_start_cols_ + i;
      int matrix_ = input_matrix_1[global_cols_ * rows_ + j];
      int vector_ = input_vector_1[global_cols_];
      local_result_vector_[j] += matrix_ * vector_;
    }
  }

  boost::mpi::reduce(world, local_result_vector_.data(), rows_, result_vector_.data(), std::plus<>(), 0);

  return true;
}

bool sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int* ans = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(result_vector_.begin(), result_vector_.end(), ans);
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