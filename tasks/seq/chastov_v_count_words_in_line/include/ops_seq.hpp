#pragma once

#include <algorithm>
#include <cstring>
#include <iterator>
#include <sstream>

#include "core/task/include/task.hpp"

namespace chastov_v_count_words_in_line_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<char> inputString;
  int wordsFound{};
  int spacesFound{};
};

}  // namespace chastov_v_count_words_in_line_seq