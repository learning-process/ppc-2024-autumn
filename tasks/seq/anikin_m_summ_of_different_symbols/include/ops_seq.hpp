// Copyright 2024 Anikin Maksim
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace anikin_m_sum_of_differnt_symbols_seq {

class SumDifSymSequential : public ppc::core::Task {
 public:
  explicit SumDifSymSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::string> input;
  int res;
};

}  // namespace anikin_m_sum_of_differnt_symbols_seq