#include "seq/kovalev_k_bubble_sort_oddeven_transposition/include/header.hpp"

template <class T>
bool kovalev_k_bubble_sort_oddeven_transposition_seq::BubbleSortOddEvenTransposition<T>::bubble_sort(T* arr,
                                                                                                     size_t length) {
  for (int i = 0; i < length - 1; i++)
    for (int j = 0; j < length - i - 1; j++)
      if (arr[j] > arr[j + 1]) std::swap(arr[j], arr[j + 1]);
  return true;
}

template <class T>
bool kovalev_k_bubble_sort_oddeven_transposition_seq::BubbleSortOddEvenTransposition<T>::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == n && taskData->inputs_count[0] == n);
}

template <class T>
bool kovalev_k_bubble_sort_oddeven_transposition_seq::BubbleSortOddEvenTransposition<T>::pre_processing() {
  internal_order_test();
  v = std::vector<T>(n);
  void* ptr_input = taskData->inputs[0];
  void* ptr_vec = v.data();
  memcpy(ptr_vec, ptr_input, sizeof(T) * n);
  return true;
}

template <class T>
bool kovalev_k_bubble_sort_oddeven_transposition_seq::BubbleSortOddEvenTransposition<T>::run() {
  internal_order_test();
  return bubble_sort(v.data(), n);
}

template <class T>
bool kovalev_k_bubble_sort_oddeven_transposition_seq::BubbleSortOddEvenTransposition<T>::post_processing() {
  internal_order_test();
  memcpy(reinterpret_cast<T*>(taskData->outputs[0]), v.data(), sizeof(T) * n);
  return true;
}

template class kovalev_k_bubble_sort_oddeven_transposition_seq::BubbleSortOddEvenTransposition<int>;
template class kovalev_k_bubble_sort_oddeven_transposition_seq::BubbleSortOddEvenTransposition<double>;