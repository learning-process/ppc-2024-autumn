#include "mpi/malyshev_v_lent_horizontal/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/vector.hpp>
#include <vector>

bool malyshev_lent_horizontal::TestTaskSequential::pre_processing() {
  internal_order_test();

  uint32_t rows = taskData->inputs_count[0];
  uint32_t cols = taskData->inputs_count[1];

  matrix_.resize(rows, std::vector<int32_t>(cols));
  vector_.resize(cols);
  result_.resize(rows);

  int32_t* data;
  for (uint32_t i = 0; i < matrix_.size(); i++) {
    data = reinterpret_cast<int32_t*>(taskData->inputs[i]);
    std::copy(data, data + cols, matrix_[i].data());
  }

  data = reinterpret_cast<int32_t*>(taskData->inputs[rows]);
  std::copy(data, data + cols, vector_.data());

  return true;
}

bool malyshev_lent_horizontal::TestTaskSequential::validation() {
  internal_order_test();

  uint32_t rows = taskData->inputs_count[0];
  uint32_t cols = taskData->inputs_count[1];
  uint32_t vector_size = taskData->inputs_count[2];

  if (taskData->inputs.size() != rows + 1 || taskData->inputs_count.size() < 3) {
    return false;
  }

  if (cols != vector_size) {
    return false;
  }

  return taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool malyshev_lent_horizontal::TestTaskSequential::run() {
  internal_order_test();

  for (uint32_t i = 0; i < matrix_.size(); i++) {
    result_[i] = 0;
    for (uint32_t j = 0; j < vector_.size(); j++) {
      result_[i] += matrix_[i][j] * vector_[j];
    }
  }

  return true;
}

bool malyshev_lent_horizontal::TestTaskSequential::post_processing() {
  internal_order_test();

  std::copy(result_.begin(), result_.end(), reinterpret_cast<int32_t*>(taskData->outputs[0]));

  return true;
}

bool malyshev_lent_horizontal::TestTaskParallel::validation() {
  internal_order_test();

  uint32_t rows = taskData->inputs_count[0];
  uint32_t cols = taskData->inputs_count[1];
  uint32_t vector_size = taskData->inputs_count[2];

  if (taskData->inputs.size() != rows + 1 || taskData->inputs_count.size() < 3) {
    return false;
  }

  if (cols != vector_size) {
    return false;
  }

  return taskData->outputs_count[0] == rows;
}

bool malyshev_lent_horizontal::TestTaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    uint32_t rows = taskData->inputs_count[0];
    uint32_t cols = taskData->inputs_count[1];

    delta_ = rows / world.size();
    ext_ = rows % world.size();

    matrix_.resize(rows, std::vector<int32_t>(cols));
    vector_.resize(cols);
    result_.resize(rows);

    int32_t* data;
    for (uint32_t i = 0; i < matrix_.size(); i++) {
      data = reinterpret_cast<int32_t*>(taskData->inputs[i]);
      std::copy(data, data + cols, matrix_[i].data());
    }

    data = reinterpret_cast<int32_t*>(taskData->inputs[rows]);
    std::copy(data, data + cols, vector_.data());
  }

  return true;
}

bool malyshev_lent_horizontal::TestTaskParallel::run() {
  internal_order_test();

  broadcast(world, delta_, 0);
  broadcast(world, ext_, 0);
  broadcast(world, vector_, 0);

  if (world.rank() == 0) {
    flat_matrix_.clear();
    for (const auto& row : matrix_) {
      flat_matrix_.insert(flat_matrix_.end(), row.begin(), row.end());
    }
  }

  std::vector<int32_t> sendcounts(world.size(), delta_ * vector_.size());
  for (uint32_t i = 0; i < ext_; ++i) {
    sendcounts[world.size() - i - 1] += vector_.size();
  }

  std::vector<int32_t> displs(world.size(), 0);
  for (size_t i = 1; i < displs.size(); ++i) {
    displs[i] = displs[i - 1] + sendcounts[i - 1];
  }

  uint32_t local_rows = sendcounts[world.rank()] / vector_.size();
  local_matrix_.resize(local_rows, std::vector<int32_t>(vector_.size()));
  std::vector<int32_t> flat_local_matrix(local_rows * vector_.size());

  scatterv(world, flat_matrix_.data(), sendcounts, displs, flat_local_matrix.data(), 0);

  for (uint32_t i = 0; i < local_rows; ++i) {
    std::copy(flat_local_matrix.begin() + i * vector_.size(), flat_local_matrix.begin() + (i + 1) * vector_.size(),
              local_matrix_[i].begin());
  }

  local_result_.resize(local_rows);
  for (uint32_t i = 0; i < local_matrix_.size(); i++) {
    local_result_[i] = 0;
    for (uint32_t j = 0; j < vector_.size(); j++) {
      local_result_[i] += local_matrix_[i][j] * vector_[j];
    }
  }

  std::vector<int32_t> recvcounts = sendcounts;
  gatherv(world, local_result_.data(), recvcounts, displs, result_.data(), 0);

  return true;
}

bool malyshev_lent_horizontal::TestTaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    std::copy(result_.begin(), result_.end(), reinterpret_cast<int32_t*>(taskData->outputs[0]));
  }

  return true;
}