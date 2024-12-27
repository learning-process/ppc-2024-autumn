#include "seq/tyurin_m_count_sentences_in_string/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool tyurin_m_count_sentences_in_string_seq::SentenceCountTaskSequential::pre_processing() {
  internal_order_test();
  input_str_ = *reinterpret_cast<std::string*>(taskData->inputs[0]);
  sentence_count_ = 0;
  return true;
}

bool tyurin_m_count_sentences_in_string_seq::SentenceCountTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
}

bool tyurin_m_count_sentences_in_string_seq::SentenceCountTaskSequential::run() {
  internal_order_test();

  bool inside_sentence = false;

  for (char c : input_str_) {
    if (is_sentence_end(c)) {
      if (inside_sentence) {
        sentence_count_++;
        inside_sentence = false;
      }
    } else if (!is_whitespace(c)) {
      inside_sentence = true;
    }
  }

  return true;
}

bool tyurin_m_count_sentences_in_string_seq::SentenceCountTaskSequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<int*>(taskData->outputs[0]) = sentence_count_;
  return true;
}

bool tyurin_m_count_sentences_in_string_seq::SentenceCountTaskSequential::is_sentence_end(char c) {
  return c == '.' || c == '!' || c == '?';
}

bool tyurin_m_count_sentences_in_string_seq::SentenceCountTaskSequential::is_whitespace(char c) {
  return c == ' ' || c == '\n' || c == '\t';
}