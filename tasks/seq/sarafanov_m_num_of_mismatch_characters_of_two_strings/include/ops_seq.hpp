#pragma once

#include <string>

#include "core/task/include/task.hpp"

namespace sarafanov_m_num_of_mismatch_characters_of_two_strings_seq {

class SequentialTask : public ppc::core::Task {
 public:
  explicit SequentialTask(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string input_a_, input_b_;
  int result_{};
};

}  // namespace sarafanov_m_num_of_mismatch_characters_of_two_strings_seq
