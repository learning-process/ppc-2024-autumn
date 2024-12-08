#include "mpi/sedova_o_vertical_ribbon_scheme/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

bool sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI::validation() {
  internal_order_test();
  if (!taskData) {
    return false;
  }
  if (taskData->inputs[0] == nullptr || taskData->inputs[1] == nullptr) {
    return false;
  }
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 &&
         taskData->inputs_count[0] % taskData->inputs_count[1] == 0;
}

bool sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI::pre_processing() {
  internal_order_test();
    int* matrix = reinterpret_cast<int*>(taskData->inputs[0]);
    int* vector = reinterpret_cast<int*>(taskData->inputs[1]);

    int count = taskData->inputs_count[0];
    rows_ = taskData->inputs_count[1];
    cols_ = count / rows_;

    input_matrix_1.assign(matrix, matrix + count);
    input_vector_1.assign(vector, vector + rows_);
    result_vector_.resize(cols_, 0);

    proc.resize(world.size(), 0);
    off.resize(world.size(), -1);

    if (world.size() > rows_) {
      for (int i = 0; i < rows_; ++i) {
        off[i] = i * cols_;
        proc[i] = cols_;
      }
      for (int i = rows_; i < world.size(); ++i) {
        off[i] = -1;
        proc[i] = 0;
      }
    } else {
      int count_proc = rows_ / world.size();
      int surplus = rows_ % world.size();
      int offset = 0;
      for (int i = 0; i < world.size(); ++i) {
        if (surplus > 0) {
          proc[i] = (count_proc + 1) * cols_;
          --surplus;
        } else {
          proc[i] = count_proc * cols_;
        }
        off[i] = offset;
        offset += proc[i];
      }
    }
  return true;
}

bool sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI::run() {
  internal_order_test();
  boost::mpi::broadcast(world, rows_, 0);
  boost::mpi::broadcast(world, cols_, 0);
  boost::mpi::broadcast(world, proc, 0);
  boost::mpi::broadcast(world, off, 0);
  boost::mpi::broadcast(world, input_matrix_1, 0);
  boost::mpi::broadcast(world, input_vector_1, 0);
  int proc_start = off[world.rank()] / cols_;
  int matrix_start_ = proc[world.rank()] / cols_;
  std::vector<int> proc_result(cols_, 0);

  for (int i = 0; i < cols_; ++i) {
    for (int j = 0; j < matrix_start_; ++j) {
      int prog_start = proc_start + j;
      if (prog_start < rows_) {
        int matrix = input_matrix_1[i * rows_ + prog_start];
        int vector = input_vector_1[prog_start];
        proc_result[j] += matrix * vector;
      }
    }
  }

  boost::mpi::reduce(world, proc_result.data(), cols_, result_vector_.data(), std::plus<>(), 0);

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
  if (!taskData) {
    return false;
  }
  if (taskData->inputs[0] == nullptr || taskData->inputs[1] == nullptr) {
    return false;
  }
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 &&
         taskData->inputs_count[0] % taskData->inputs_count[1] == 0;
}

bool sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI::pre_processing() {
  internal_order_test();

  matrix_ = reinterpret_cast<int*>(taskData->inputs[0]);
  vector_ = reinterpret_cast<int*>(taskData->inputs[1]);
  count = taskData->inputs_count[0];
  rows_ = taskData->inputs_count[1];
  cols_ = count / rows_;
  result_vector_.assign(cols_, 0);

  return true;
}

bool sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI::run() {
  internal_order_test();

  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      result_vector_[j] += matrix_[i * cols_ + j] * vector_[i];
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