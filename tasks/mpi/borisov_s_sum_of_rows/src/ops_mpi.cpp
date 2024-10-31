// Copyright 2023 Nesterov Alexander
#include "mpi/borisov_s_sum_of_rows/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool borisov_s_sum_of_rows::SumOfRowsTaskSequential ::pre_processing() {
  internal_order_test();

  size_t rows = taskData->inputs_count[0];
  size_t cols = taskData->inputs_count[1];

  if (rows > 0 && cols > 0) {
    matrix_.resize(rows, std::vector<int>(cols));
    int* data = reinterpret_cast<int*>(taskData->inputs[0]);
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        matrix_[i][j] = data[(i * cols) + j];
      }
    }
  }

  row_sums_.resize(rows, 0);
  return true;
}

bool borisov_s_sum_of_rows::SumOfRowsTaskSequential::validation() {
  internal_order_test();

  if (taskData->outputs_count[0] != taskData->inputs_count[0]) {
    return false;
  }

  size_t cols = taskData->inputs_count.size() > 1 ? taskData->inputs_count[1] : 0;
  if (cols <= 0) {
    return false;
  }

  if (!taskData->inputs[0] || !taskData->outputs[0]) {
    return false;
  }

  return true;
}

bool borisov_s_sum_of_rows::SumOfRowsTaskSequential ::run() {
  internal_order_test();

  if (!matrix_.empty() && !matrix_[0].empty()) {
    for (size_t i = 0; i < matrix_.size(); i++) {
      int row_sum = 0;
      for (size_t j = 0; j < matrix_[i].size(); j++) {
        row_sum += matrix_[i][j];
      }
      row_sums_[i] = row_sum;
    }
  }
  return true;
}

bool borisov_s_sum_of_rows::SumOfRowsTaskSequential ::post_processing() {
  internal_order_test();

  if (!row_sums_.empty()) {
    int* out = reinterpret_cast<int*>(taskData->outputs[0]);
    for (size_t i = 0; i < row_sums_.size(); i++) {
      out[i] = row_sums_[i];
    }
  }
  return true;
}

std::vector<int> borisov_s_sum_of_rows::getRandomMatrix(size_t rows, size_t cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> matrix(rows * cols);
  for (auto& element : matrix) {
    element = static_cast<int>(gen() % 100);
  }
  return matrix;
}

bool borisov_s_sum_of_rows::SumOfRowsTaskParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;

  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();

    size_t rows = taskData->inputs_count[0];
    size_t cols = taskData->inputs_count.size() > 1 ? taskData->inputs_count[1] : 0;

    if (rows > 0 && cols > 0) {
      int* data = reinterpret_cast<int*>(taskData->inputs[0]);
      matrix_.resize(rows * cols);

      for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
          matrix_[i * cols + j] = data[(i * cols) + j];
        }
      }

      for (size_t pr = 1; pr < world.size(); pr++) {
        int size = static_cast<int>(delta * cols);
        world.send(pr, 0, matrix_.data() + (pr * delta * cols), size);
      }
    }
  }

  broadcast(world, delta, 0);

  loc_matrix_.resize(delta * taskData->inputs_count[1]);
  if (world.rank() == 0) {
    loc_matrix_.assign(matrix_.begin(), matrix_.begin() + (delta * taskData->inputs_count[1]));
  } else {
    int size = static_cast<int>(delta * taskData->inputs_count[1]);
    world.recv(0, 0, loc_matrix_.data(), size);
  }

  loc_row_sums_.resize(delta, 0);
  res = 0;
  return true;
}

bool borisov_s_sum_of_rows::SumOfRowsTaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->outputs_count[0] != taskData->inputs_count[0]) {
      return false;
    }

    size_t cols = taskData->inputs_count.size() > 1 ? taskData->inputs_count[1] : 0;
    if (cols <= 0) {
      return false;
    }

    if (!taskData->inputs[0] || !taskData->outputs[0]) {
      return false;
    }
  }
  return true;
}

bool borisov_s_sum_of_rows::SumOfRowsTaskParallel::run() {
  internal_order_test();

  for (size_t i = 0; i < loc_row_sums_.size(); i++) {
    loc_row_sums_[i] = 0;
    for (size_t j = 0; j < taskData->inputs_count[1]; j++) {
      loc_row_sums_[i] += loc_matrix_[i * taskData->inputs_count[1] + j];
    }
  }

  if (world.rank() == 0) {
    row_sums_.resize(taskData->inputs_count[0], 0);
  }

  reduce(world, loc_row_sums_.data(), loc_row_sums_.size(), row_sums_.data(), std::plus<>(), 0);

  return true;
}

bool borisov_s_sum_of_rows::SumOfRowsTaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    if (!row_sums_.empty()) {
      int* out = reinterpret_cast<int*>(taskData->outputs[0]);
      for (size_t i = 0; i < row_sums_.size(); i++) {
        out[i] = row_sums_[i];
      }
    }
  }
  return true;
}
