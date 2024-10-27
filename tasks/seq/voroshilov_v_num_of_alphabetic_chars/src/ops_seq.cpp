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
  for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = ptr[i];
  }
  res_ = 0;
  return true;
}

bool voroshilov_v_num_of_alphabetic_chars_seq::AlphabetCharsTaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < input_.size(); i++) {
    int code = input_[i];
    if (((code >= 65) && (code <= 90)) || ((code >= 97) && (code <= 122))) //ASCII codes of english alphabet
      res_++;
  }
  return true;
}

bool voroshilov_v_num_of_alphabetic_chars_seq::AlphabetCharsTaskSequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<int*>(taskData->outputs[0]) = res_;
  return true;
}
