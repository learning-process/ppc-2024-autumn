#pragma once

#include <string>

#include "core/task/include/task.hpp"

namespace tyurin_m_count_sentences_in_string_seq {

class SentenceCountTaskSequential : public ppc::core::Task {
 public:
  explicit SentenceCountTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string input_str_;
  int sentence_count_ = 0;

  static bool is_sentence_end(char c);
  static bool is_whitespace(char c);
};

}  // namespace tyurin_m_count_sentences_in_string_seq
