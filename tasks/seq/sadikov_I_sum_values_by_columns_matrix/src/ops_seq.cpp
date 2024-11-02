#include <iostream>

#include "seq/sadikov_I_sum_values_by_columns_matrix/include/sq_task.h"

using namespace std::chrono_literals;

std::vector<double> sadikov_I_Sum_values_by_columns_matrix_seq::Randvector(size_t size) {
  srand(time(nullptr));
  std::vector<double> vec(size);
  for (size_t i = 0; i < size; ++i) {
    vec[i] = static_cast<int>(rand()) / static_cast<int>(size * 100000);
  }
  return vec;
}

sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask::MatrixTask(std::shared_ptr<ppc::core::TaskData> td)
    : Task(std::move(td)) {}

bool sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask::validation() {
  internal_order_test();
  return taskData->inputs_count[1] == taskData->outputs_count[0];
}

bool sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask::pre_processing() {
  internal_order_test();
  rows_count = static_cast<size_t>(taskData->inputs_count[0]);
  columns_count = static_cast<size_t>(taskData->inputs_count[1]);
  matrix.reserve(rows_count * columns_count);
  auto *tmp_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
  for (size_t i = 0; i < columns_count; ++i) {
    for (size_t j = 0; j < rows_count; ++j) {
      matrix.emplace_back(tmp_ptr[j * columns_count + i]);
    }
  }
  sum.reserve(rows_count);
  return true;
}

bool sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask::run() {
  internal_order_test();
  calculate(columns_count);
  return true;
}

bool sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < columns_count; ++i) {
    reinterpret_cast<int *>(taskData->outputs[0])[i] = sum[i];
  }
  return true;
}

void sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask::calculate(size_t size) {
  for (size_t i = 0; i < size; ++i) {
    sum[i] = std::accumulate(matrix.begin() + static_cast<int>(i * rows_count),
                             matrix.begin() + static_cast<int>((i + 1) * rows_count), 0);
  }
}

std::shared_ptr<ppc::core::TaskData> sadikov_I_Sum_values_by_columns_matrix_seq::CreateTaskData(
    std::vector<int> &InV, const std::vector<int> &CeV, std::vector<int> &OtV) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(InV.data()));
  taskData->inputs_count.emplace_back(CeV[0]);
  taskData->inputs_count.emplace_back(CeV[1]);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(OtV.data()));
  taskData->outputs_count.emplace_back(OtV.size());
  return taskData;
}