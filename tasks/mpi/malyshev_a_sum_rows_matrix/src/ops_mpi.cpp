#include "mpi/malyshev_a_sum_rows_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/vector.hpp>
#include <vector>

bool malyshev_a_sum_rows_matrix_mpi::TestTaskSequential ::pre_processing() {
  internal_order_test();

  uint32_t rows = taskData->inputs_count[0];
  uint32_t cols = taskData->inputs_count[1];

  input_.resize(rows, std::vector<int32_t>(cols));
  res_.resize(rows);

  int32_t* data;
  for (uint32_t i = 0; i < input_.size(); i++) {
    data = reinterpret_cast<int32_t*>(taskData->inputs[i]);
    std::copy(data, data + cols, input_[i].data());
  }

  return true;
}

bool malyshev_a_sum_rows_matrix_mpi::TestTaskSequential::validation() {
  internal_order_test();

  return taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool malyshev_a_sum_rows_matrix_mpi::TestTaskSequential ::run() {
  internal_order_test();

  for (uint32_t i = 0; i < input_.size(); i++) {
    res_[i] = std::accumulate(input_[i].begin(), input_[i].end(), 0);
  }

  return true;
}

bool malyshev_a_sum_rows_matrix_mpi::TestTaskSequential ::post_processing() {
  internal_order_test();

  std::copy(res_.begin(), res_.end(), reinterpret_cast<int32_t*>(taskData->outputs[0]));

  return true;
}

bool malyshev_a_sum_rows_matrix_mpi::TestTaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    uint32_t rows = taskData->inputs_count[0];
    uint32_t cols = taskData->inputs_count[1];

    delta_ = rows / world.size();
    ext_ = rows % world.size();

    input_.resize(rows, std::vector<int32_t>(cols));
    res_.resize(rows);

    int32_t* data;
    for (uint32_t i = 0; i < input_.size(); i++) {
      data = reinterpret_cast<int32_t*>(taskData->inputs[i]);
      std::copy(data, data + cols, input_[i].data());
    }
  }

  return true;
}

bool malyshev_a_sum_rows_matrix_mpi::TestTaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return taskData->outputs_count[0] == taskData->inputs_count[0];
  }

  return true;
}

bool malyshev_a_sum_rows_matrix_mpi::TestTaskParallel::run() {
  internal_order_test();

  broadcast(world, delta_, 0);
  broadcast(world, ext_, 0);

  std::vector<int32_t> sizes(world.size(), delta_);
  for (uint32_t i = 0; i < ext_; i++) {
    sizes[world.size() - i - 1]++;
  }

  local_input_.resize(sizes[world.rank()]);
  local_res_.resize(sizes[world.rank()]);

  scatterv(world, input_, sizes, local_input_.data(), 0);

  for (uint32_t i = 0; i < local_input_.size(); i++) {
    local_res_[i] = std::accumulate(local_input_[i].begin(), local_input_[i].end(), 0);
  }

  gatherv(world, local_res_, res_.data(), sizes, 0);

  return true;
}

bool malyshev_a_sum_rows_matrix_mpi::TestTaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    std::copy(res_.begin(), res_.end(), reinterpret_cast<int32_t*>(taskData->outputs[0]));
  }

  return true;
}
