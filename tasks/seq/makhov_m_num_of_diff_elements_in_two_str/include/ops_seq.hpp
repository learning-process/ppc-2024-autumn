// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace makhov_m_num_of_diff_elements_in_two_str_seq {

int countDiffElem(const std::string& str1_, const std::string& str2_);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string str1{};
  std::string str2{};
  int res{};
};

}  // namespace makhov_m_num_of_diff_elements_in_two_str_seq