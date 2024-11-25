#include "seq/konkov_i_task_dining_philosophers/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool konkov_i_task_dining_philosophers::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = reinterpret_cast<int*>(taskData->inputs[0])[0];
  res = 0;
  return true;
}

bool konkov_i_task_dining_philosophers::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
}

bool konkov_i_task_dining_philosophers::TestTaskSequential::run() {
  internal_order_test();
  for (int i = 0; i < input_; i++) {
    res++;
  }
  return true;
}

bool konkov_i_task_dining_philosophers::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}