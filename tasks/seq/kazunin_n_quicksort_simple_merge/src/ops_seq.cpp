#include "seq/kazunin_n_quicksort_simple_merge/include/ops_seq.hpp"

#include <stack>

namespace kazunin_n_quicksort_simple_merge_seq {

void iterative_quicksort(std::vector<int>& data) {
  std::stack<std::pair<int, int>> stack;
  stack.emplace(0, data.size() - 1);

  while (!stack.empty()) {
    auto [low, high] = stack.top();
    stack.pop();

    if (low < high) {
      int pivot = data[high];
      int i = low - 1;

      for (int j = low; j < high; ++j) {
        if (data[j] < pivot) {
          std::swap(data[++i], data[j]);
        }
      }
      std::swap(data[i + 1], data[high]);
      int p = i + 1;

      stack.emplace(low, p - 1);
      stack.emplace(p + 1, high);
    }
  }
}

bool QuicksortSimpleMergeSeq::validation() {
  internal_order_test();

  return taskData->inputs_count[0] > 0;
}

bool QuicksortSimpleMergeSeq::pre_processing() {
  internal_order_test();

  auto* vec_data = reinterpret_cast<int*>(taskData->inputs[0]);
  int vec_size = taskData->inputs_count[0];

  input_vector.assign(vec_data, vec_data + vec_size);

  return true;
}

bool QuicksortSimpleMergeSeq::run() {
  internal_order_test();

  iterative_quicksort(input_vector);

  return true;
}

bool QuicksortSimpleMergeSeq::post_processing() {
  internal_order_test();

  auto* out_vector = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(input_vector.begin(), input_vector.end(), out_vector);

  return true;
}

}  // namespace kazunin_n_quicksort_simple_merge_seq
