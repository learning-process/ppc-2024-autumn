#include "seq/konkov_i_count_words/include/ops_seq.hpp"

#include <random>
#include <regex>
#include <sstream>

bool konkov_i_count_words_seq::CountWordsTaskSequential::pre_processing() {
  internal_order_test();
  input_ = *reinterpret_cast<std::string*>(taskData->inputs[0]);
  word_count_ = 0;
  return true;
}

bool konkov_i_count_words_seq::CountWordsTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
}

bool konkov_i_count_words_seq::CountWordsTaskSequential::run() {
  internal_order_test();
  std::regex word_regex("\\b\\w+\\b");
  auto words_begin = std::sregex_iterator(input_.begin(), input_.end(), word_regex);
  auto words_end = std::sregex_iterator();
  word_count_ = std::distance(words_begin, words_end);
  return true;
}

bool konkov_i_count_words_seq::CountWordsTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = word_count_;
  return true;
}

std::string konkov_i_count_words_seq::CountWordsTaskSequential::generate_random_string(int num_words, int word_length) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 25);

  std::string word(word_length, 'a');
  std::ostringstream oss;
  for (int i = 0; i < num_words; ++i) {
    for (int j = 0; j < word_length; ++j) {
      word[j] = 'a' + dis(gen);
    }
    oss << word << " ";
  }
  return oss.str();
}