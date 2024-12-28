// Copyright 2023 Nesterov Alexander
#pragma once

#include <algorithm>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "core/task/include/task.hpp"

namespace shlyakov_m_ccs_mult {

struct SparseMatrix {
  std::vector<double> values;
  std::vector<int> row_indices;
  std::vector<int> col_pointers;
};

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {
    std::srand(std::time(nullptr));
  }
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  SparseMatrix A_;
  int rows_a;
  int cols_a;
  SparseMatrix B_;
  int rows_b;
  int cols_b;
  SparseMatrix result_;
};

}  // namespace shlyakov_m_ccs_mult
