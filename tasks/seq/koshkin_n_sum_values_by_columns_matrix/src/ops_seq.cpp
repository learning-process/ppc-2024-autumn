// Copyright 2024 Nesterov Alexander
#include "seq/koshkin_n_sum_values_by_columns_matrix/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool koshkin_n_sum_values_by_columns_matrix_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output

  rows = taskData->inputs_count[0];
  columns = taskData->inputs_count[1];

  // Извлечение данных из TaskData
  input_.resize(rows, std::vector<int>(columns));

  // Заполнение матрицы из входных данных
  uint8_t* inputMatrix = taskData->inputs[0];  // Указатель на матрицу
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < columns; ++j) {
      input_[i][j] = reinterpret_cast<int*>(inputMatrix)[i * columns + j];
    }
  }
  res.resize(columns, 0); //sumColumns
  return true;
}

bool koshkin_n_sum_values_by_columns_matrix_seq::TestTaskSequential::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() < 1 || taskData->outputs_count.size() < 1) {
    return false;
  }

  // Проверка размера входных данных
  rows = reinterpret_cast<int*>(taskData->inputs[0])[0];
  columns = reinterpret_cast<int*>(taskData->inputs[1])[0];
  return rows > 0 && columns > 0 && taskData->inputs_count[0] == 2 && taskData->outputs_count[0] == columns;
}

bool koshkin_n_sum_values_by_columns_matrix_seq::TestTaskSequential::run() {
  internal_order_test();
  // Вычисление суммы по столбцам
  for (int j = 0; j < columns; ++j) {
    for (int i = 0; i < rows; ++i) {
      res[j] += input_[i][j];
    }
  }
  return true;
}

bool koshkin_n_sum_values_by_columns_matrix_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  uint8_t* outputSums = taskData->outputs[0];  
  for (int j = 0; j < columns; ++j) {
    reinterpret_cast<int*>(outputSums)[j] = res[j];
  }
  return true;
}