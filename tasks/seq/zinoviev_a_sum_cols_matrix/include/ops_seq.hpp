// Copyright 2023 Nesterov Alexander
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace zinoviev_a_sum_cols_matrix {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int input_{};
  int res{};
  std::vector<int> matrixData;  // для хранения значений матрицы
  std::vector<int> columnSums;  // для хранения сумм по столбцам
  int totalRows{};
  int totalCols{};

  // Функция для вычисления суммы по столбцам
  void calculateColumnSums();
};

}  // namespace zinoviev_a_sum_cols_matrix

