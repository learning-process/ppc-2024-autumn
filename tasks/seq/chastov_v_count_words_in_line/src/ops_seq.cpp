#include "seq/chastov_v_count_words_in_line/include/ops_seq.hpp"

namespace chastov_v_count_words_in_line_seq {

std::vector<char> createString(int n) {
  std::vector<char> wordCountInput;
  std::string firstSentence = "Hello my name is Slava. Now I am a third year student at Lobachevsky University. ";
  for (int i = 0; i < n - 1; i++) {
    for (unsigned long int j = 0; j < firstSentence.length(); j++) {
      wordCountInput.push_back(firstSentence[j]);
    }
  }
  std::string lastSentence = "This is a proposal to evaluate the performance of a word counting algorithm via MPI.";
  for (unsigned long int j = 0; j < lastSentence.length(); j++) {
    wordCountInput.push_back(lastSentence[j]);
  }
  return wordCountInput;
}

bool chastov_v_count_words_in_line_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  inputString = std::vector<char>(taskData->inputs_count[0]);
  auto* tmp = reinterpret_cast<char*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    inputString[i] = tmp[i];
  }
  return true;
}

bool chastov_v_count_words_in_line_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool chastov_v_count_words_in_line_seq::TestTaskSequential::run() {
  internal_order_test();
  spacesFound = 0;
  wordsFound = 0;

  bool inWord = false;

  for (char c : inputString) {
    if (std::isspace(c) != 0) {
      if (inWord) {
        inWord = false;
        spacesFound++;
      }
    } else {
      if (!inWord) {
        inWord = true;
        wordsFound++;
      }
    }
  }

  return true;
}

bool chastov_v_count_words_in_line_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = wordsFound;
  return true;
}

}  // namespace chastov_v_count_words_in_line_seq