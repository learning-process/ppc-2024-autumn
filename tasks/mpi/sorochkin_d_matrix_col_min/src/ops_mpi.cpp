#include "../include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

bool sorochkin_d_matrix_col_min_mpi::TestTaskSequential::pre_processing() {
  internal_order_test();

  rows_ = taskData->inputs_count[0];
  cols_ = taskData->inputs_count[1];

  const auto* src = reinterpret_cast<int*>(taskData->inputs[0]);
  input_.assign(src, src + (rows_ * cols_));

  res_.resize(cols_, std::numeric_limits<int>::max());

  return true;
}

bool sorochkin_d_matrix_col_min_mpi::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 &&
         taskData->outputs_count[0] == taskData->inputs_count[1];
}

bool sorochkin_d_matrix_col_min_mpi::TestTaskSequential::run() {
  internal_order_test();

  for (size_t i = 0; i < cols_; i++) {
    for (size_t j = 0; j < rows_; j++) {
      res_[i] = std::min(res_[i], input_[j * cols_ + i]);
    }
  }

  return true;
}

bool sorochkin_d_matrix_col_min_mpi::TestTaskSequential::post_processing() {
  internal_order_test();
  std::copy(res_.begin(), res_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  return true;
}

bool sorochkin_d_matrix_col_min_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    rows_ = taskData->inputs_count[0];
    cols_ = taskData->inputs_count[1];

    const auto* src = reinterpret_cast<int*>(taskData->inputs[0]);

    input_.resize(rows_ * cols_);
    for (size_t i = 0; i < rows_; i++) {
      for (size_t j = 0; j < cols_; j++) {
        input_[j * rows_ + i] = src[i * cols_ + j];
      }
    }

    res_.resize(cols_);
  }

  return true;
}

bool sorochkin_d_matrix_col_min_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() != 0) {
    return true;
  }

  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 &&
         taskData->outputs_count[0] == taskData->inputs_count[1];
}

bool sorochkin_d_matrix_col_min_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  boost::mpi::broadcast(world, rows_, 0);
  boost::mpi::broadcast(world, cols_, 0);

  const int remainder = cols_ % world.size();
  std::vector<int> sizes(world.size(), cols_ / world.size());
  for (int i = 0; i < remainder; i++) {
    ++sizes[i];
  }

  const size_t local_cols_n = sizes[world.rank()];
  std::vector<int> local_input(local_cols_n * rows_);
  std::vector<int> local_res(local_cols_n, std::numeric_limits<int>::max());

  std::for_each(sizes.begin(), sizes.end(), [&](auto& d) { d *= rows_; });
  boost::mpi::scatterv(world, input_, sizes, local_input.data(), 0);

  for (size_t i = 0; i < local_cols_n; i++) {
    for (size_t j = 0; j < rows_; j++) {
      local_res[i] = std::min(local_res[i], local_input[i * rows_ + j]);
    }
  }

  std::for_each(sizes.begin(), sizes.end(), [&](auto& d) { d /= rows_; });
  boost::mpi::gatherv(world, local_res, res_.data(), sizes, 0);

  return true;
}

bool sorochkin_d_matrix_col_min_mpi::TestMPITaskParallel::post_processing() {
  world.barrier();
  internal_order_test();
  if (world.rank() == 0) {
    std::copy(res_.begin(), res_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }
  return true;
}
