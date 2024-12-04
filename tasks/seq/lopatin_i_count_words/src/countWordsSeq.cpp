#include "seq/lopatin_i_count_words/include/countWordsSeqHeader.hpp"

namespace lopatin_i_count_words_seq {

std::vector<char> generateLongString(int n) {
  std::vector<char> testData;
  std::string testString = "This is a long sentence for performance testing of the word count algorithm using MPI. ";
  for (int i = 0; i < n - 1; i++) {
    for (unsigned long int j = 0; j < testString.length(); j++) {
      testData.push_back(testString[j]);
    }
  }
  std::string lastSentence = "This is a long sentence for performance testing of the word count algorithm using MPI.";
  for (unsigned long int j = 0; j < lastSentence.length(); j++) {
    testData.push_back(lastSentence[j]);
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
  return true;
}

bool lopatin_i_count_words_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool lopatin_i_count_words_seq::TestTaskSequential::run() {
  internal_order_test();

  wordCount = 0;
  bool inWord = false;

  for (size_t i = 0; i < input_.size(); ++i) {
    if (input_[i] == ' ') {
      if (inWord) {
        wordCount++;
        inWord = false;
      }
    } else if (!inWord) {
      inWord = true;
    }
  }

  if (inWord) {
    wordCount++;
  }

  return true;
}

bool lopatin_i_count_words_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = wordCount;
  return true;
}

}  // namespace lopatin_i_count_words_seq