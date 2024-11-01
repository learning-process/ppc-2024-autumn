// Copyright 2024 Nesterov Alexander
#include "seq/zinoviev_a_sum_cols_matrix/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool zinoviev_a_sum_cols_matrix::TestTaskSequential::pre_processing() {
  internal_order_test();

  // ������ �������� �������
  totalRows = reinterpret_cast<int*>(taskData->inputs[0])[0];
  totalCols = reinterpret_cast<int*>(taskData->inputs[0])[1];

  // ������������� �������� ��� �������� �������� ������� � ���� �� ��������
  matrixData.resize(totalRows * totalCols);
  columnSums.resize(totalCols, 0);

  // ������ �������� ������� �� ������� ������
  for (int i = 0; i < totalRows * totalCols; i++) {
    matrixData[i] = reinterpret_cast<int*>(taskData->inputs[0])[i + 2];
  }

  return true;
}

bool zinoviev_a_sum_cols_matrix::TestTaskSequential::validation() {
  internal_order_test();
  // �������� ���������� ��������� ������� � �������� ������
  return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
}

bool zinoviev_a_sum_cols_matrix::TestTaskSequential::run() {
  internal_order_test();
  // ���������� ����� �� ��������
  calculateColumnSums();
  return true;
}

bool zinoviev_a_sum_cols_matrix::TestTaskSequential::post_processing() {
  internal_order_test();

  // ������ ����������� ����� �� �������� � �������� ������
  for (int j = 0; j < totalCols; j++) {
    reinterpret_cast<int*>(taskData->outputs[0])[j] = columnSums[j];
  }

  return true;
}

void zinoviev_a_sum_cols_matrix::TestTaskSequential::calculateColumnSums() {
  // ������ �� ���� ��������� ������� � ���������� ���� �� ��������
  for (int i = 0; i < totalRows; i++) {
    for (int j = 0; j < totalCols; j++) {
      columnSums[j] += matrixData[i * totalCols + j];
    }
  }
}