// Copyright 2024 Nesterov Alexander
#include "seq/zaytsev_d_num_of_alternations_signs/include/ops_seq.hpp"

bool zaytsev_d_num_of_alternations_signs_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  int* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
  int input_count = taskData->inputs_count[0];
  data_.assign(input_data, input_data + input_count);
  res = 0;
  return true;
}

bool zaytsev_d_num_of_alternations_signs_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool zaytsev_d_num_of_alternations_signs_seq::TestTaskSequential::run() {
  internal_order_test();
  if (data_.size() < 2) {
    res = 0;
    return true;
  }

  for (size_t i = 1; i < data_.size(); ++i) {
    if ((data_[i] >= 0 && data_[i - 1] < 0) || (data_[i] < 0 && data_[i - 1] >= 0)) {
      res++;
    }
  }

  return true;
}

bool zaytsev_d_num_of_alternations_signs_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
