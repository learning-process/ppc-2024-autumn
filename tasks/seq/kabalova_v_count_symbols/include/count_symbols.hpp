// Copyright 2024 Kabalova Valeria
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace kabalova_v_count_symbols_seq {

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

}  // namespace kabalova_v_count_symbols_seq