#include "mpi/sotskov_a_sum_element_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <chrono>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool sotskov_a_sum_element_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  auto* tmp_ptr_matrix = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* tmp_ptr_rows = reinterpret_cast<int*>(taskData->inputs[1]);
  auto* tmp_ptr_cols = reinterpret_cast<int*>(taskData->inputs[2]);
  matrix_.assign(tmp_ptr_matrix, tmp_ptr_matrix + (*tmp_ptr_rows) * (*tmp_ptr_cols));
  rows_ = *tmp_ptr_rows;
  cols_ = *tmp_ptr_cols;
  return true;
}

bool sotskov_a_sum_element_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool sotskov_a_sum_element_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  result_ = std::accumulate(matrix_.begin(), matrix_.end(), 0.0);
  return true;
}

bool sotskov_a_sum_element_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<double*>(taskData->outputs[0]) = result_;
  return true;
}

bool sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* tmp_ptr_matrix = reinterpret_cast<double*>(taskData->inputs[0]);
    auto* tmp_ptr_rows = reinterpret_cast<int*>(taskData->inputs[1]);
    auto* tmp_ptr_cols = reinterpret_cast<int*>(taskData->inputs[2]);
    matrix_.assign(tmp_ptr_matrix, tmp_ptr_matrix + (*tmp_ptr_rows) * (*tmp_ptr_cols));
    rows_ = *tmp_ptr_rows;
    cols_ = *tmp_ptr_cols;
  }
  broadcast(world, rows_, 0);
  broadcast(world, cols_, 0);
  if (world.rank() != 0) {
    matrix_.resize(rows_ * cols_);
  }
  broadcast(world, matrix_.data(), matrix_.size(), 0);
  return true;
}

bool sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  local_result_ = parallel_sum_elements(matrix_);
  reduce(world, local_result_, global_result_, std::plus<>(), 0);
  return true;
}

bool sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = global_result_;
  }
  return true;
}

double sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel::parallel_sum_elements(const std::vector<double>& matrix) {
  int rank = world.rank();
  int size = world.size();
  double local_sum = 0.0;
  int elements_per_process = matrix.size() / size;
  int start_idx = rank * elements_per_process;
  int end_idx = (rank == size - 1) ? matrix.size() : start_idx + elements_per_process;
  for (int i = start_idx; i < end_idx; ++i) {
    local_sum += matrix[i];
  }
  return local_sum;
}