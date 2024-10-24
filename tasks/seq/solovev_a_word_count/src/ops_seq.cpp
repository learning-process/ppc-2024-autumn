#include "seq/solovev_a_word_count/include/ops_seq.hpp"

namespace solovev_a_word_count_seq {

std::vector<char> create_text(int quan_words) {
  std::vector<char> res;
  std::string word = "word ";
  std::string last = "word.";
  for (int i = 0; i < quan_words - 1; i++)
    for (unsigned long int symbol = 0; symbol < word.length(); symbol++) {
      res.push_back(word[symbol]);
    }
  for (unsigned long int symbol = 0; symbol < last.length(); symbol++) {
    res.push_back(last[symbol]);
  }
  return res;
}

bool solovev_a_word_count_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<char>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  res = 0;
  return true;
}

bool solovev_a_word_count_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool solovev_a_word_count_seq::TestTaskSequential::run() {
  internal_order_test();
  for (char symbol : input_) {
    if (symbol != ' ' && symbol != '.') {
    } else {
      res++;
    }
  }
  return true;
}

bool solovev_a_word_count_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

}  // namespace solovev_a_word_count_seq
