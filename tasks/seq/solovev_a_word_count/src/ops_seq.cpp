#include "seq/solovev_a_word_count/include/ops_seq.hpp"

namespace solovev_a_word_count_seq {

bool solovev_a_word_count_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<char>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
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
    if (symbol == ' ' || symbol == '.') {
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
