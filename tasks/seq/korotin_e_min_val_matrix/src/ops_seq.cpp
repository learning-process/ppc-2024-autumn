// Copyright 2024 Nesterov Alexander
#include "seq/korotin_e_min_val_matrix/include/ops_seq.hpp"

#include <algorithm>
#include <thread>

using namespace std::chrono_literals;

std::vector<double> korotin_e_min_val_matrix_seq::getRandomMatrix(const unsigned rows, const unsigned columns, double scal) {
  if (rows == 0 || columns == 0) {
    throw std::invalid_argument("Can't creaate matrix with 0 rows or columns");
  }

  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<double> matrix(rows * columns);
  for (unsigned i = 0; i < rows * columns; i++) {
    matrix[i] = gen() / scal;
  }
  return matrix;
}

bool korotin_e_min_val_matrix_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = std::vector<double>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  // Init value for output
  res = 0.0;
  return true;
}

bool korotin_e_min_val_matrix_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1;
}

bool korotin_e_min_val_matrix_seq::TestTaskSequential::run() {
  internal_order_test();
  res = *std::min_element(input_.begin(), input_.end());
  std::this_thread::sleep_for(20ms);
  return true;
}

bool korotin_e_min_val_matrix_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}
