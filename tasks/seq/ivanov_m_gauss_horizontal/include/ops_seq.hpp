// Copyright 2024 Ivanov Mike
#pragma once

#include <cmath>
#include <vector>

#include "core/task/include/task.hpp"

#define DELTA 1e-9

namespace ivanov_m_gauss_horizontal_seq {
int get_linear_index(int row, int col, int number_of_columns);
void swap_rows(std::vector<double>& matrix, int first_row, int second_row, int number_of_columns);
int find_max_row(const std::vector<double>& matrix, int source_row, int source_column, int number_of_rows,
                 int number_of_columns);
int determinant(const std::vector<double>& matrix, int size);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> extended_matrix;  // extended matrix
  int number_of_equations;
  std::vector<double> res;
};

}  // namespace ivanov_m_gauss_horizontal_seq