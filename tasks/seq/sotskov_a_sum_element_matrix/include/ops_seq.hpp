// Copyright 2024 Sotskov Andrey
#pragma once
#include <vector>
#include <memory>
#include "core/task/include/task.hpp"

namespace sotskov_a_sum_element_matrix_seq {

class TestTaskSequential : public ppc::core::Task {
  public:
    explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
    bool pre_processing() override;
    bool validation() override;
    bool run() override;
    bool post_processing() override;

  private:
    std::vector<std::vector<int>> matrix_;
    int result_;
  };
} // namespace sotskov_a_sum_element_matrix_seq
