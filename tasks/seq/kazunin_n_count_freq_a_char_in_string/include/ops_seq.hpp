// Copyright 2023 Nesterov Alexander
#pragma once
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace kazunin_n_count_freq_a_char_in_string_seq {

class CountFreqCharTaskSequential : public ppc::core::Task {
 public:
  explicit CountFreqCharTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool run() override;
  bool validation() override;
  bool pre_processing() override;
  bool post_processing() override;

 private:
  char target_character_{};
  int frequency_count_ = 0;
  std::string input_string_;
};
}  // namespace kazunin_n_count_freq_a_char_in_string_seq
