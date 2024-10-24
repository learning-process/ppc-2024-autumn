// Copyright 2024 Nesterov Alexander
#include "seq/gnitienko_k_sum_values_by_rows_matrix/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool gnitienko_k_sum_row_seq::SumByRowSeq::pre_processing() {
  internal_order_test();

  rows = taskData->inputs_count[0]; 
  cols = taskData->inputs_count[1];  

  input_.resize(rows * cols);
  auto* ptr = reinterpret_cast<int*>(taskData->inputs[0]);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      input_[i * cols + j] = ptr[i * cols + j]; 
    }
  }

  res = std::vector<int>(rows, 0); 

  return true;
}

bool gnitienko_k_sum_row_seq::SumByRowSeq::validation() {
  internal_order_test();
  // Check count elements of output
  return (taskData->inputs_count.size() == 2 && taskData->inputs_count[0] >= 0 && taskData->inputs_count[1] >= 0 &&
          taskData->outputs_count[0] == taskData->inputs_count[0]);
}

std::vector<int> gnitienko_k_sum_row_seq::SumByRowSeq::mainFunc() {
  for (int i = 0; i < rows; ++i) {
    int sum = 0;
    for (int j = 0; j < cols; ++j) {
      sum += input_[i * cols + j];  // Индексируем вектор как двумерный
    }
    res[i] = sum;  // Сохраняем сумму в выходной вектор
  }
  return res;
}

bool gnitienko_k_sum_row_seq::SumByRowSeq::run() {
  internal_order_test();
  mainFunc();
  std::this_thread::sleep_for(20ms);
  return true;
}

bool gnitienko_k_sum_row_seq::SumByRowSeq::post_processing() {
  internal_order_test();
  //reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  std::memcpy(taskData->outputs[0], res.data(), rows * sizeof(int));
  return true;
}
