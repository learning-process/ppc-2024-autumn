#include "mpi/sedova_o_vertical_ribbon_scheme/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <thread>
#include <vector>

bool sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    // Check for null TaskData
    if (!taskData) {
      return false;
    }
    // Check input counts
    if (taskData->inputs_count[0] < 1 || taskData->inputs_count[1] < 1) {
      return false;
    }
    // Check if the number of columns in the matrix is equal to the size of the vector.
    int num_rows = taskData->inputs_count[0] / taskData->inputs_count[1];
    if (taskData->inputs_count[0] % taskData->inputs_count[1] != nullptr || num_rows < 1) {
      return false;
    }
    // Check for null input pointers
    if (taskData->inputs[0] == nullptr || taskData->inputs[1] == nullptr) {
      return false;
    }
    // Check output
    if (taskData->outputs[0] == nullptr) {
      return false;
    }
  }
  return true;
}
bool sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
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
  else {
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
    int* answer = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(result_vector_.begin(), result_vector_.end(), answer);
  }
  return true;
}

bool sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI::validation() {
  internal_order_test();
  return taskData->inputs_count[0] >= 1 && taskData->inputs_count[1] >= 1 && !taskData &&
         taskData->inputs_count[0] % taskData->inputs_count[1] == nullptr && taskData->outputs[0] != nullptr;
 }

bool sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI::pre_processing() {
  internal_order_test();

  input_matrix_ = reinterpret_cast<int*>(taskData->inputs[0]);
  input_vector_ = reinterpret_cast<int*>(taskData->inputs[1]);
  int count = taskData->inputs_count[0];
  num_rows_ = taskData->inputs_count[1];
  num_cols_ = count / num_rows_;
  result_vector_.assign(num_cols_, 0);

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