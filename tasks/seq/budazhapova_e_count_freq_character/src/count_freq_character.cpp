#include <thread>

#include "seq/budazhapova_e_count_freq_character/include/count_freq_character_header.hpp"

using namespace std::chrono_literals;

bool budazhapova_e_count_freq_character_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_ = *reinterpret_cast<std::string*>(taskData->inputs[0]);
  symb = input_[0];
  res = 0;
  return true;
}

bool budazhapova_e_count_freq_character_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 1 && (taskData->outputs_count[0] == 1);
}

bool budazhapova_e_count_freq_character_seq::TestTaskSequential::run() {
  internal_order_test();
  for (unsigned long i = 0; i < input_.length(); i++) {
    if (input_[i] == symb) {
      res++;
    }
  }
  return true;
}

bool budazhapova_e_count_freq_character_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0]) = res;
  return true;
}
