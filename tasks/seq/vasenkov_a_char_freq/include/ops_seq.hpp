#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace vasenkov_a_char_frequency_seq {

class CharFrequencyTaskSequential : public ppc::core::Task {
 public:
  explicit CharFrequencyTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  char target_char_;
  int frequency_ = 0;
  std::string str_input_;
};

}  // namespace vasenkov_a_char_frequency_seq
