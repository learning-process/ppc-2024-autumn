#include "seq/chastov_v_bubble_sort/include/ops_seq.hpp"

#include <algorithm>

using namespace chastov_v_bubble_sort;

template <class T>
bool TestTaskSequential<T>::bubble_sort(T* mass, size_t len) {
  for (size_t i = 0; i < len - 1; i++) {
    for (size_t j = 0; j < len - i - 1; j++) {
      if (mass[j] > mass[j + 1]) {
        std::swap(mass[j], mass[j + 1]);
      }
    }
  }
  return true;
}

template <class T>
bool TestTaskSequential<T>::pre_processing() {
  internal_order_test();
  data = std::vector<T>(data_size);
  T* input_data = reinterpret_cast<T*>(taskData->inputs[0]);
  std::copy(input_data, input_data + data_size, data.begin());
  return true;
}

template <class T>
bool TestTaskSequential<T>::validation() {
  internal_order_test();
  size_t num_inputs = taskData->inputs_count[0];
  size_t num_outputs = taskData->outputs_count[0];
  bool is_valid = (num_inputs > 0) && (num_outputs == data_size) && (num_inputs == data_size);
  return is_valid;
}

template <class T>
bool TestTaskSequential<T>::run() {
  internal_order_test();
  return bubble_sort(data.data(), data_size);
}

template <class T>
bool TestTaskSequential<T>::post_processing() {
  internal_order_test();
  std::copy(data.begin(), data.end(), reinterpret_cast<T*>(taskData->outputs[0]));
  return true;
}

namespace chastov_v_bubble_sort {
template class TestTaskSequential<int>;
template class TestTaskSequential<double>;
}  // namespace chastov_v_bubble_sort