#include "seq/kharin_m_number_of_sentences_seq/include/ops_seq.hpp"

#include <string>

namespace kharin_m_number_of_sentences_seq {

bool CountSentencesSequential::pre_processing() {
  internal_order_test();
  text = std::string(reinterpret_cast<const char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  sentence_count = 0;
  return true;
}

bool CountSentencesSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool CountSentencesSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < text.size(); i++) {
    char c = text[i];
    if (c == '.' || c == '?' || c == '!') {
      sentence_count++;
    }
  }
  return true;
}

bool CountSentencesSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = sentence_count;
  return true;
}

}  // namespace kharin_m_number_of_sentences_seq
