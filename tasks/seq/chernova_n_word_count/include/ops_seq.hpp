#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace chernova_n_word_count_seq {

std::vector<char> clean_string(const std::vector<char>& input);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<char> input_;
  int spaceCount;
};

}  // namespace chernova_n_word_count_seq