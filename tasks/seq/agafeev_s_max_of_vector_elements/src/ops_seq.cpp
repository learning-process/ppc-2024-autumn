#include "seq/agafeev_s_max_of_vector_elements/include/ops_seq.hpp"

namespace agafeev_s_max_of_vector_elements_seq {

/*
template <typename T>
std::vector<T> create_RandomMatrix(int row_size, int column_size) {
  auto rand_gen = std::mt19937(1337);
  std::vector<T> matrix(row_size * column_size);
  for (unsigned int i = 0; i < matrix.size(); i++) matrix[i] = rand_gen() % 100;

  return matrix;
}
*/

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
