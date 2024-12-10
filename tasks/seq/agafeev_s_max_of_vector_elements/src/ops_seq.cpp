#include "seq/agafeev_s_max_of_vector_elements/include/ops_seq.hpp"

namespace agafeev_s_max_of_vector_elements_seq {

template <typename T>
T get_MaxValue(std::vector<T> matrix) {
  T max_result = std::numeric_limits<T>::min();
  for (unsigned int i = 0; i < matrix.size(); i++)
    if (max_result < matrix[i]) max_result = matrix[i];

  return max_result;
}

template <typename T>
bool MaxMatrixSequental<T>::pre_processing() {
  internal_order_test();

  // Init value
  auto* temp_ptr = reinterpret_cast<T*>(taskData->inputs[0]);
  input_.insert(input_.begin(), temp_ptr, temp_ptr + taskData->inputs_count[0]);

  return true;
}

template <typename T>
bool MaxMatrixSequental<T>::validation() {
  internal_order_test();

  return taskData->outputs_count[0] == 1;
}

template <typename T>
bool MaxMatrixSequental<T>::run() {
  internal_order_test();

  maxres_ = get_MaxValue(input_);

  return true;
}

template <typename T>
bool MaxMatrixSequental<T>::post_processing() {
  internal_order_test();

  reinterpret_cast<T*>(taskData->outputs[0])[0] = maxres_;

  return true;
}

template class MaxMatrixSequental<int>;

}  // namespace agafeev_s_max_of_vector_elements_seq
