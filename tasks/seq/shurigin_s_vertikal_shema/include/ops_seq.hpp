#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shurigin_s_vertikal_shema {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int* matrix_data_{nullptr};                   // Указатель на исходную матрицу
  std::vector<std::vector<int>> input_matrix_;  // Матрица как вектор столбцов
  std::vector<int> input_vector_;               // Вектор-строка
  std::vector<int> result_vector_;              // Результирующий вектор-строка
  int num_rows_{0};
  int num_cols_{0};
};

}  // namespace shurigin_s_vertikal_shema