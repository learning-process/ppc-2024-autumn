#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace voroshilov_v_num_of_alphabetic_chars_seq {

class AlphabetCharsTaskSequential : public ppc::core::Task {
 public:
  explicit AlphabetCharsTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<char> input_;
  int res_;
};

}  // namespace voroshilov_v_num_of_alphabetic_chars_seq