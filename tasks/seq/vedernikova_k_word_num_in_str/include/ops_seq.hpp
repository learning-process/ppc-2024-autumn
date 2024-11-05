#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace vedernikova_k_word_num_in_str_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string input_{};
  size_t res_{};
};

}  // namespace vedernikova_k_word_num_in_str_seq