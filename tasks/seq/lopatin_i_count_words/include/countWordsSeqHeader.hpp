#pragma once

#include <string>

#include "core/task/include/task.hpp"

namespace lopatin_i_count_words_seq {

int countWords(const std::string& str);
std::string generateLongString(int n);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string input_;
  int wordCount{};
};

}  // namespace lopatin_i_count_words_seq