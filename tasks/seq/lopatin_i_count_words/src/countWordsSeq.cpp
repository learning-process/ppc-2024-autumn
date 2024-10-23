#include "seq/lopatin_i_count_words/include/countWordsSeqHeader.hpp"

namespace lopatin_i_count_words_seq {

int countWords(const std::string& str) {
  std::istringstream iss(str);
  return std::distance(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>());
}

std::string generateLongString(int n) {
  std::string testData;
  std::string testString = "This is a long sentence for performance testing of the word count algorithm using MPI. ";
  for (int i = 0; i < n; i++) {
    testData += testString;
  }
  return testData;
}

bool lopatin_i_count_words_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init string
  input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  // Init value for output
  wordCount = 0;
  return true;
}

bool lopatin_i_count_words_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check if inout is not empty and output is prepared
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool lopatin_i_count_words_seq::TestTaskSequential::run() {
  internal_order_test();
  wordCount = countWords(input_);
  return true;
}

bool lopatin_i_count_words_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = wordCount;
  return true;
}

}  // namespace lopatin_i_count_words_seq