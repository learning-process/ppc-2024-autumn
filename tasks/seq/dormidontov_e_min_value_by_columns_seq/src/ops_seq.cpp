#include "seq/dormidontov_e_min_value_by_columns_seq/include/ops_seq.hpp"

#include <climits>
#include <thread>
bool dormidontov_e_min_value_by_columns_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  rs = taskData->inputs_count[0];
  cs = taskData->inputs_count[1];

  input_.resize(rs, std::vector<int>(cs));

  for (int i = 0; i < rs; ++i) {
    for (int j = 0; j < cs; ++j) {
      input_[i][j] = reinterpret_cast<int*>(taskData->inputs[0])[i * (cs) + j];
    }
  }

  res.resize(cs, 0);
  return true;
}

bool dormidontov_e_min_value_by_columns_seq::TestTaskSequential::validation() {
  internal_order_test();
  return ((taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0) &&
          taskData->inputs_count[1] == taskData->outputs_count[0]);
}

bool dormidontov_e_min_value_by_columns_seq::TestTaskSequential::run() {
  internal_order_test();
  for (int j = 0; j < cs; ++j) {
    res[j] = INT_MAX;
    for (int i = 0; i < rs; ++i) {
      if (res[j] > input_[i][j]) {
        res[j] = input_[i][j];
      }
    }
  }
  return true;
}

bool dormidontov_e_min_value_by_columns_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < cs; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}
