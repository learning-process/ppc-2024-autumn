// Copyright 2024 Nesterov Alexander
#include "seq/leontev_n_gather/include/ops_seq.hpp"

template <class InOutType>
bool leontev_n_mat_vec_seq::MatVecSequential<InOutType>::pre_processing() {
  internal_order_test();
  InOutType* vec_ptr1;
  InOutType* vec_ptr2;
  if (taskData->inputs.size() >= 2) {
    vec_ptr1 = reinterpret_cast<InOutType*>(taskData->inputs[0]);
    vec_ptr2 = reinterpret_cast<InOutType*>(taskData->inputs[1]);
  } else {
    return false;
  }
  mat_ = std::vector<InOutType>(taskData->inputs_count[0]);
  vec_ = std::vector<InOutType>(taskData->inputs_count[1]);
  for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
    mat_[i] = vec_ptr1[i];
  }
  for (size_t i = 0; i < taskData->inputs_count[1]; i++) {
    vec_[i] = vec_ptr2[i];
  }
  res = std::vector<InOutType>(vec_.size(), InOutType(0));
  return true;
}

template <class InOutType>
bool leontev_n_mat_vec_seq::MatVecSequential<InOutType>::validation() {
  internal_order_test();
  // Matrix+Vector input && vector input
  if (taskData->inputs.size() != 2 || taskData->outputs.size() != 1) {
    return false;
  }
  // square matrix
  if (taskData->inputs_count[0] != taskData->inputs_count[1] * taskData->inputs_count[1]) {
    return false;
  }
  if (taskData->inputs_count[0] == 0) {
    return false;
  }
  return true;
}

template <class InOutType>
bool leontev_n_mat_vec_seq::MatVecSequential<InOutType>::run() {
  internal_order_test();
  for (size_t i = 0; i < res.size(); i++) {
    for (size_t j = 0; j < res.size(); j++) {
      res[i] += mat_[i * res.size() + j] * vec_[j];
    }
  }
  return true;
}

template <class InOutType>
bool leontev_n_mat_vec_seq::MatVecSequential<InOutType>::post_processing() {
  internal_order_test();
  std::copy(res.begin(), res.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  return true;
}

template class leontev_n_mat_vec_seq::MatVecSequential<int32_t>;
template class leontev_n_mat_vec_seq::MatVecSequential<uint32_t>;
template class leontev_n_mat_vec_seq::MatVecSequential<float>;
template class leontev_n_mat_vec_seq::MatVecSequential<double>;
