#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "core/task/include/task.hpp"

namespace kharin_m_number_of_sentences_seq {

class CountSentencesSequential : public ppc::core::Task {
 public:
  explicit CountSentencesSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string text;
  int sentence_count{};
};
}  // namespace kharin_m_number_of_sentences_seq
