#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace guseynov_e_check_lex_order_of_two_string_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<char>> input_;
  int res_{};
};

}  // namespace guseynov_e_check_lex_order_of_two_string_seq