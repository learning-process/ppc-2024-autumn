// Copyright 2023 Konkov Ivan
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace konkov_i_count_words_seq {

class CountWordsTaskSequential : public ppc::core::Task {
 public:
  explicit CountWordsTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string input_;
  int word_count_{};
};

}  // namespace konkov_i_count_words_seq