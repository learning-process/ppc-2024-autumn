#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace shpynov_n_amount_of_mismatched_numbers_seq {
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::string> input_{};  // two input strings
  int result{};
};
}  // namespace shpynov_n_amount_of_mismatched_numbers_seq