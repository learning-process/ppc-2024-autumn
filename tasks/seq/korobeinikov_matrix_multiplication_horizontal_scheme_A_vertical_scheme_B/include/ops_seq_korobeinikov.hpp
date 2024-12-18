// Copyright 2024 Korobeinikov Arseny
#pragma once

#include <cassert>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace korobeinikov_a_test_task_seq_lab_02 {

struct Matrix {
  std::vector<int> data;
  int count_rows;
  int count_cols;

  Matrix() {
    count_rows = 0;
    count_cols = 0;
    data = std::vector<int>();
  }

  Matrix(int count_rows_, int count_cols_) {
    count_rows = count_rows_;
    count_cols = count_cols_;
    data = std::vector<int>(count_cols * count_rows);
  }
  int& get_el(int row, int col) {
    size_t index = row * count_cols + col;
    assert(index < data.size());
    return data[index];
  }

  const int& get_el(int row, int col) const {
    size_t index = row * count_cols + col;
    assert(index < data.size());
    return data[index];
  }
};

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  Matrix A;
  Matrix B;

  Matrix res;
};

}  // namespace korobeinikov_a_test_task_seq_lab_02