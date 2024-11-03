#include "gtest/gtest.h"
#include "seq/konkov_i_count_words/include/ops_seq.hpp"

namespace konkov_i_count_words_seq {

TEST(konkov_i_count_words_seq, test_simple) {
  std::string input = "Hello world";
  int result = countWords(input);
  EXPECT_EQ(result, 2);
}

TEST(konkov_i_count_words_seq, test_empty) {
  std::string input =;
  int result = countWords(input);
  EXPECT_EQ(result, 0);
}

TEST(konkov_i_count_words_seq, test_single_word) {
  std::string input = "Hello";
  int result = countWords(input);
  EXPECT_EQ(result, 1);
}

TEST(konkov_i_count_words_seq, test_multiple_spaces) {
  std::string input = "Hello   world";
  int result = countWords(input);
  EXPECT_EQ(result, 2);
}

TEST(konkov_i_count_words_seq, test_leading_spaces) {
  std::string input = "   Hello world";
  int result = countWords(input);
  EXPECT_EQ(result, 2);
}

TEST(konkov_i_count_words_seq, test_trailing_spaces) {
  std::string input = "Hello world   ";
  int result = countWords(input);
  EXPECT_EQ(result, 2);
}

}  // namespace konkov_i_count_words_seq
