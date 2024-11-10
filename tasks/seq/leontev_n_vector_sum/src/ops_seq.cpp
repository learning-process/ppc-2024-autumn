// Copyright 2024 Nesterov Alexander
#include "seq/leontev_n_vector_sum/include/ops_seq.hpp"

template <class InOutType>
bool leontev_n_vector_sum_seq::VecSumSequential<InOutType>::pre_processing() {
  internal_order_test();
  input_ = std::vector<InOutType>(taskData->inputs_count[0]);
  auto* vec_ptr = reinterpret_cast<InOutType*>(taskData->inputs[0]);
  for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = vec_ptr[i];
  }
  res = 0;
  return true;
}

template <class InOutType>
bool leontev_n_vector_sum_seq::VecSumSequential<InOutType>::validation() {
  internal_order_test();
  // Input vector exists and output is a single number
  return taskData->inputs_count[0] != 0 && taskData->outputs_count[0] == 1;
}

template <class InOutType>
bool leontev_n_vector_sum_seq::VecSumSequential<InOutType>::run() {
  internal_order_test();
  res = std::accumulate(input_.begin(), input_.end(), 0);
  return true;
}

template <class InOutType>
bool leontev_n_vector_sum_seq::VecSumSequential<InOutType>::post_processing() {
  internal_order_test();
  reinterpret_cast<InOutType*>(taskData->outputs[0])[0] = res;
  return true;
}

template class leontev_n_vector_sum_seq::VecSumSequential<int32_t>;
template class leontev_n_vector_sum_seq::VecSumSequential<uint32_t>;
template class leontev_n_vector_sum_seq::VecSumSequential<float>;
template class leontev_n_vector_sum_seq::VecSumSequential<double>;
