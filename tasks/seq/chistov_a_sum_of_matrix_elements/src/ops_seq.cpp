#include "seq/chistov_a_sum_of_matrix_elements/include/ops_seq.hpp"

namespace chistov_a_sum_of_matrix_elements_seq {

template <typename T>
bool TestTaskSequential<T>::pre_processing() {
  internal_order_test();

  T* tmp_ptr = reinterpret_cast<T*>(taskData->inputs[0]);
  input_.assign(tmp_ptr, tmp_ptr + taskData->inputs_count[0]);
  return true;
}

template <typename T>
bool TestTaskSequential<T>::validation() {
  internal_order_test();

  return taskData->outputs_count[0] == 1;
}

template <typename T>
bool TestTaskSequential<T>::run() {
  internal_order_test();

  res = std::accumulate(input_.begin(), input_.end(), 0);
  return true;
}

template <typename T>
bool TestTaskSequential<T>::post_processing() {
  internal_order_test();

  if (!taskData->outputs.empty() && taskData->outputs[0] != nullptr) {
    reinterpret_cast<T*>(taskData->outputs[0])[0] = res;
    return true;
  }
  return false;
}

template class TestTaskSequential<int>;
template class TestTaskSequential<double>;

}  // namespace chistov_a_sum_of_matrix_elements_seq