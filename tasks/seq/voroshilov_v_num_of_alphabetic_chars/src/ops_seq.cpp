#include "seq/voroshilov_v_num_of_alphabetic_chars/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool voroshilov_v_num_of_alphabetic_chars_seq::AlphabetCharsTaskSequential::validation() {
  internal_order_test();
  // Check count elements of input and output
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool voroshilov_v_num_of_alphabetic_chars_seq::AlphabetCharsTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = std::vector<char>(taskData->inputs_count[0]);
  char* ptr = reinterpret_cast<char*>(taskData->inputs[0]);
  std::copy(ptr, ptr + taskData->inputs_count[0], input_.begin());
  res_ = 0;
  return true;
}

bool voroshilov_v_num_of_alphabetic_chars_seq::AlphabetCharsTaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < input_.size(); i++) {
    if (std::isalpha(input_[i]) != 0) {  // Check if it is alphabetic character
      res_++;
    }
  }
  return true;
}

bool voroshilov_v_num_of_alphabetic_chars_seq::AlphabetCharsTaskSequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<int*>(taskData->outputs[0]) = res_;
  return true;
}
