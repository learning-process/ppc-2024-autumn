// Copyright 2024 Nesterov Alexander
#include "seq/beresnev_a_cannons_algorithm/include/ops_seq.hpp"

#include <algorithm>
#include <string>
#include <vector>

bool beresnev_a_cannons_algorithm_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  auto* A = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* B = reinterpret_cast<double*>(taskData->inputs[1]);
  inp_A.assign(A, A + s_);
  inp_B.assign(B, B + s_);
  res_.resize(s_);
  return true;
}

bool beresnev_a_cannons_algorithm_seq::TestTaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs[2] != nullptr) {
    n_ = reinterpret_cast<size_t*>(taskData->inputs[2])[0];
  }
  s_ = n_ * n_;
  return n_ > 0 && taskData->inputs_count[2] == 1 && taskData->inputs_count[0] == taskData->inputs_count[1] &&
         taskData->inputs_count[1] == s_ && taskData->inputs[0] != nullptr && taskData->inputs[1] != nullptr &&
         taskData->outputs[0] != nullptr && static_cast<size_t>(taskData->outputs_count[0]) == s_;
}

bool beresnev_a_cannons_algorithm_seq::TestTaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < n_; ++i) {
    for (size_t j = 0; j < n_; ++j) {
      for (size_t k = 0; k < n_; ++k) {
        res_[i * n_ + j] += inp_A[i * n_ + k] * inp_B[k * n_ + j];
      }
    }
  }
  return true;
}

bool beresnev_a_cannons_algorithm_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<std::vector<double>*>(taskData->outputs[0])[0].assign(res_.data(), res_.data() + s_);
  return true;
}