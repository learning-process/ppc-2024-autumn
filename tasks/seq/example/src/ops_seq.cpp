// Copyright 2024 Nesterov Alexander
#include "seq/example/include/ops_seq.hpp"

#include <random>
#include <thread>

using namespace std::chrono_literals;

std::vector<int> nesterov_a_test_task_seq::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool nesterov_a_test_task_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  // Init value for output
  count_rows = (int)*taskData->inputs[1];
  size_rows = (int)(taskData->inputs_count[0] / (*taskData->inputs[1]));
  res = std::vector<int>(count_rows, 0);
  return true;
}

bool nesterov_a_test_task_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return *taskData->inputs[1] == taskData->outputs_count[0];
}

bool nesterov_a_test_task_seq::TestTaskSequential::run() {
  internal_order_test();  
  for (int i = 0; i < count_rows; i++) {
    res[i] = *std::min_element(input_.begin() + i * size_rows, input_.begin() + (i + 1) * size_rows);
  }
  return true;
}

bool nesterov_a_test_task_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < count_rows; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}
