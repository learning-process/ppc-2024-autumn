#include "mpi/sedova_o_vertical_ribbon_scheme/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <thread>
#include <vector>

bool sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 1 && taskData->inputs_count[1] > 0 &&
         taskData->inputs_count[0] % taskData->inputs_count[1] == 0 &&
         taskData->outputs_count[0] == taskData->inputs_count[0] / taskData->inputs_count[1];
}

bool sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI::pre_processing() {
  internal_order_test();
  matrix_ = reinterpret_cast<int*>(taskData->inputs[0]);
  vector_ = reinterpret_cast<int*>(taskData->inputs[1]);
  int count = taskData->inputs_count[0];
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

bool sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 &&
         taskData->inputs_count[0] % taskData->inputs_count[1] == 0 &&
         taskData->outputs_count[0] == taskData->inputs_count[0] / taskData->inputs_count[1];
}

bool sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    if (!taskData || taskData->inputs[0] == nullptr || taskData->inputs[1] == nullptr ||
        taskData->outputs[0] == nullptr) {
      return false;
    }

    int* input_A = reinterpret_cast<int*>(taskData->inputs[0]);
    int* input_B = reinterpret_cast<int*>(taskData->inputs[1]);

    int count = taskData->inputs_count[0];
    rows_ = taskData->inputs_count[1];
    cols_ = count / rows_;

    input_matrix_1.assign(input_A, input_A + count);
    input_vector_1.assign(input_B, input_B + rows_);
    result_vector_.resize(cols_, 0);

    proc.resize(world.size(), 0);
    off.resize(world.size(), -1);

    if (world.size() > rows_) {
      for (int i = 0; i < rows_; ++i) {
        proc[i] = cols_;
        off[i] = i * cols_;
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

  for (int i = 0; i < matrix_start_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      int prog_start = proc_start + i;
      int matr = input_matrix_1[cols_ * prog_start + j];
      int vec = input_vector_1[prog_start];
      proc_result[j] += matr * vec;
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