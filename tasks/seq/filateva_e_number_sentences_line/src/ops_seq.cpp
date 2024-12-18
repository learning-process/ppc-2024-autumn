// Filateva Elizaveta Number_of_sentences_per_line

#include "seq/filateva_e_number_sentences_line/include/ops_seq.hpp"

#include <thread>

bool filateva_e_number_sentences_line_seq::NumberSentencesLine::pre_processing() {
  internal_order_test();
  // Init value for input and output
  line = std::string(std::move(reinterpret_cast<char*>(taskData->inputs[0])));
  sentence_count = 0;
  return true;
}

bool filateva_e_number_sentences_line_seq::NumberSentencesLine::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
}

bool filateva_e_number_sentences_line_seq::NumberSentencesLine::run() {
  internal_order_test();
  for (long unsigned int i = 0; i < line.size(); ++i) {
    if (line[i] == '.' || line[i] == '?' || line[i] == '!') {
      ++sentence_count;
    }
  }
  if (!line.empty() && line.back() != '.' && line.back() != '?' && line.back() != '!') {
    ++sentence_count;
  }
  return true;
}

bool filateva_e_number_sentences_line_seq::NumberSentencesLine::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = sentence_count;
  return true;
}
