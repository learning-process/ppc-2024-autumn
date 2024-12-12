// Copyright 2024 Stroganov Mikhail
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace stroganov_m_count_symbols_in_string_seq {

int countSymbols(std::string& str);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string input_{};
  int result{};
};

}  // namespace stroganov_m_count_symbols_in_string_seq
