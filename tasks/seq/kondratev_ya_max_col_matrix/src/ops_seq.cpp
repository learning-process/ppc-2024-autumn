// Copyright 2024 Nesterov Alexander
#include "seq/kondratev_ya_max_col_matrix/include/ops_seq.hpp"

#include <algorithm>
#include <thread>

using namespace std::chrono_literals;

std::vector<std::vector<int32_t>> kondratev_ya_max_col_matrix_seq::getRandomMatrix(uint32_t row, uint32_t col) {
  if (row == 0 || col == 0) {
    throw std::invalid_argument("Args should be greater then zero");
  }

  int32_t low = -200;
  int32_t high = 200;

  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<std::vector<int32_t>> mtrx(row, std::vector<int32_t>(col));
  for (uint32_t i = 0; i < row; i++) {
    for (uint32_t j = 0; j < col; j++) {
      mtrx[i][j] = low + gen() % (high - low + 1);
    }
  }
  return mtrx;
}

void kondratev_ya_max_col_matrix_seq::insertRefValue(std::vector<std::vector<int32_t>>& mtrx, int32_t ref) {
  std::random_device dev;
  std::mt19937 gen(dev());

  uint32_t ind;
  uint32_t row = mtrx.size();
  uint32_t col = mtrx[0].size();

  for (uint32_t j = 0; j < col; j++) {
    ind = gen() % row;
    mtrx[ind][j] = ref;
  }
}

bool kondratev_ya_max_col_matrix_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  uint32_t row = taskData->inputs_count[0];
  uint32_t col = taskData->inputs_count[1];

  std::vector<int32_t*> tmp(row);
  for (uint32_t i = 0; i < row; i++) {
    tmp[i] = reinterpret_cast<int32_t*>(taskData->inputs[i]);
  }

  input_ = std::vector(col, std::vector<int32_t>(row));
  for (uint32_t j = 0; j < col; j++) {
    for (uint32_t i = 0; i < row; i++) {
      input_[j][i] = tmp[i][j];
    }
  }
  res_ = std::vector<int32_t>(col);

  return true;
}

bool kondratev_ya_max_col_matrix_seq::TestTaskSequential::validation() {
  internal_order_test();

  return taskData->outputs_count[0] == taskData->inputs_count[1];
}

bool kondratev_ya_max_col_matrix_seq::TestTaskSequential::run() {
  internal_order_test();

  for (uint32_t i = 0; i < input_.size(); i++) {
    res_[i] = *std::max_element(input_[i].begin(), input_[i].end());
  }

  return true;
}

bool kondratev_ya_max_col_matrix_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  auto* output_matrix = reinterpret_cast<int32_t*>(taskData->outputs[0]);
  std::copy(res_.begin(), res_.end(), output_matrix);

  return true;
}
