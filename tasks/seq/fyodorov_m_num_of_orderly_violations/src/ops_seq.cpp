// Copyright 2024 Nesterov Alexander
#include "seq/fyodorov_m_num_of_orderly_violations/include/ops_seq.hpp"

#include <algorithm>
#include <vector>
namespace fyodorov_m_num_of_orderly_violations_seq {

bool TestTaskSequential::pre_processing() {
  input_.assign(reinterpret_cast<int*>(taskData->inputs[0]),
                reinterpret_cast<int*>(taskData->inputs[0]) + taskData->inputs_count[0]);
  violations_count = 0;
  return true;
}

bool TestTaskSequential::validation() { return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1; }

/*
bool TestTaskSequential::validation() {
  return !input_.empty() && taskData->inputs_count[0]>0;
}
*/
bool TestTaskSequential::run() {
  for (size_t i = 1; i < input_.size(); ++i) {
    if (input_[i] < input_[i - 1]) {
      ++violations_count;
    }
  }
  return true;
}

bool TestTaskSequential::post_processing() {
  reinterpret_cast<int*>(taskData->outputs[0])[0] = violations_count;
  return true;
}

}  // namespace fyodorov_m_num_of_orderly_violations_seq
