// Copyright 2024 Nesterov Alexander
#include "seq/zinoviev_a_sum_cols_matrix/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool zinoviev_a_sum_cols_matrix::TestTaskSequential::pre_processing() {
  internal_order_test();

  // Чтение размеров матрицы
  totalRows = reinterpret_cast<int*>(taskData->inputs[0])[0];
  totalCols = reinterpret_cast<int*>(taskData->inputs[0])[1];

  // Инициализация векторов для хранения значений матрицы и сумм по столбцам
  matrixData.resize(totalRows * totalCols);
  columnSums.resize(totalCols, 0);

  // Чтение значений матрицы из входных данных
  for (int i = 0; i < totalRows * totalCols; i++) {
    matrixData[i] = reinterpret_cast<int*>(taskData->inputs[0])[i + 2];
  }

  return true;
}

bool zinoviev_a_sum_cols_matrix::TestTaskSequential::validation() {
  internal_order_test();
  // Проверка количества элементов входных и выходных данных
  return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
}

bool zinoviev_a_sum_cols_matrix::TestTaskSequential::run() {
  internal_order_test();
  // Вычисление суммы по столбцам
  calculateColumnSums();
  return true;
}

bool zinoviev_a_sum_cols_matrix::TestTaskSequential::post_processing() {
  internal_order_test();

  // Запись результатов суммы по столбцам в выходные данные
  for (int j = 0; j < totalCols; j++) {
    reinterpret_cast<int*>(taskData->outputs[0])[j] = columnSums[j];
  }

  return true;
}

void zinoviev_a_sum_cols_matrix::TestTaskSequential::calculateColumnSums() {
  // Проход по всем элементам матрицы и вычисление сумм по столбцам
  for (int i = 0; i < totalRows; i++) {
    for (int j = 0; j < totalCols; j++) {
      columnSums[j] += matrixData[i * totalCols + j];
    }
  }
}