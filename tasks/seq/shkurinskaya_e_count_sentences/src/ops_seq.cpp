#include "seq/shkurinskaya_e_count_sentences/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool shkurinskaya_e_count_sentences::TestTaskSequential::pre_processing() {
  internal_order_test();
  text = *reinterpret_cast<std::string*>(taskData->inputs[0]);
  res = 0;
  return true;
}

bool shkurinskaya_e_count_sentences::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool shkurinskaya_e_count_sentences::TestTaskSequential::run() {
  internal_order_test();
  bool in_end = false;
  for (size_t i = 0; i < text.size(); i++) {
    char ch = text[i];
    if (ch == '!' || ch == '?' || ch == '.') {
      if (!in_end) {
        res++;
        in_end = true;
      }
    } else if (ch != ' ') {
      in_end = false;
    }
  }
  return true;
}

bool shkurinskaya_e_count_sentences::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
