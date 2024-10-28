#include "seq/example/include/ops_seq.hpp"
#include<math.h>
#include <thread>

using namespace std::chrono_literals;

bool volochaev_s_count_characters_27_seq::Lab1_27::pre_processing() {
  internal_order_test();
  // Init value for input and output
  //input_ = reinterpret_cast<int*>(taskData->inputs[0])[0];
  input1_ = *reinterpret_cast<std::string*>(taskData->inputs[0]);
  input2_ = *reinterpret_cast<std::string*>(taskData->inputs[1]);
  res = 0;
  return true;
}

bool volochaev_s_count_characters_27_seq::Lab1_27::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 1 && taskData->inputs_count[1] == 1 && taskData->outputs_count[0] == 1;
}

bool volochaev_s_count_characters_27_seq::Lab1_27::run() {
  internal_order_test();
  res = abs((int)input1_.size() - (int)input2_.size());

  for (int i = 0; i < std::min(input1_.length(), input2_.length()); ++i)
  {
    if (input1_[i] != input2_[i]) {
      res += 2;
    }
  }

  return true;
}

bool volochaev_s_count_characters_27_seq::Lab1_27::post_processing() {
  internal_order_test();
  *reinterpret_cast<int*>(taskData->outputs[0]) = res;
  return true;
}
