#include "seq/lopatin_i_count_words/include/countWordsSeqHeader.hpp"

namespace lopatin_i_count_words_seq {

std::vector<char> generateLongString(int n) {
  std::vector<char> testData;
  std::string testString = "This is a long sentence for performance testing of the word count algorithm using MPI. ";
  for (int i = 0; i < n; i++) {
    for (unsigned long int j = 0; j < testString.length(); j++) {
      testData.push_back(testString[j]);
    }
  }
  return testData;
}

bool lopatin_i_count_words_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<char>(taskData->inputs_count[0]);
  auto* tempPtr = reinterpret_cast<char*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tempPtr[i];
  }
  wordCount = 0;
  return true;
}

bool lopatin_i_count_words_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool lopatin_i_count_words_seq::TestTaskSequential::run() {
  internal_order_test();
  bool inWord = false;
  for (char c : input_) {
    if (c == ' ' || c == '\n') {
      inWord = false;
    } else if (!inWord) {
      wordCount++;
      inWord = true;
    }
  }
  return true;
}

bool lopatin_i_count_words_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = wordCount;
  return true;
}

}  // namespace lopatin_i_count_words_seq