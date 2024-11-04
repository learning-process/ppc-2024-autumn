#pragma once

#include <algorithm>
#include <string>

#include "core/task/include/task.hpp"

namespace burykin_m_word_count {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  static bool is_word_character(char c);

 private:
  std::string input_;
  int word_count_{};
  static int count_words(const std::string& text);
};

}  // namespace burykin_m_word_count
