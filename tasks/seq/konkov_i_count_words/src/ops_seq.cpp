#include "seq/konkov_i_count_words/include/ops_seq.hpp"

#include <sstream>

namespace konkov_i_count_words_seq {

int countWords(const std::string& input) {
  std::istringstream stream(input);
  std::string word;
  int count = 0;
  while (stream >> word) {
    ++count;
  }
  return count;
}

}  // namespace konkov_i_count_words_seq