#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace strakhov_a_str_char_freq_seq {

class TaskStringCharactersFrequencySequential : public ppc::core::Task {
 public:
  explicit TaskStringCharactersFrequencySequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  char target_;
  float res = 0;
  std::string input_;
};

}  // namespace strakhov_a_str_char_freq_seq