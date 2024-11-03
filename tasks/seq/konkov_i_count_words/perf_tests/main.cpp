#include "gtest/gtest.h"
#include "seq/konkov_i_count_words/include/ops_seq.hpp"

namespace konkov_i_count_words_seq {

TEST(konkov_i_count_words_seq, test_performance) {
  std::string input = "Lorem ipsum dolor sit amet, consectetur adipiscing elit.";
  for (int i = 0; i < 1000000; ++i) {
    countWords(input);
  }
}

TEST(konkov_i_count_words_seq, test_performance_large_input) {
  std::string input;
  for (int i = 0; i < 1000; ++i) {
    input += "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ";
  }
  for (int i = 0; i < 10000; ++i) {
    countWords(input);
  }
}

}  // namespace konkov_i_count_words_seq