#include "seq/kapustin_i_bubble/include/avg_seq.hpp"

bool kapustin_i_bubble_sort_seq::BubbleSortSequential::pre_processing() {
  internal_order_test();
  int total_elements = taskData->inputs_count[0];
  auto* raw_data = reinterpret_cast<int*>(taskData->inputs[0]);
  input_.assign(raw_data, raw_data + total_elements);
  return true;
}

bool kapustin_i_bubble_sort_seq::BubbleSortSequential::validation() {
  internal_order_test();
  return !(taskData->inputs.empty() || taskData->inputs_count.empty());
}

bool kapustin_i_bubble_sort_seq::BubbleSortSequential::run() {
  internal_order_test();
  bool swapped;
  int n = input_.size();

  do {
    swapped = false;

    for (int i = 1; i < n - 1; i += 2) {
      if (input_[i] > input_[i + 1]) {
        std::swap(input_[i], input_[i + 1]);
        swapped = true;
      }
    }

    for (int i = 0; i < n - 1; i += 2) {
      if (input_[i] > input_[i + 1]) {
        std::swap(input_[i], input_[i + 1]);
        swapped = true;
      }
    }
  } while (swapped);

  return true;
}
bool kapustin_i_bubble_sort_seq::BubbleSortSequential::post_processing() {
  internal_order_test();
  std::memcpy(taskData->outputs[0], input_.data(), input_.size() * sizeof(int));

  return true;
}