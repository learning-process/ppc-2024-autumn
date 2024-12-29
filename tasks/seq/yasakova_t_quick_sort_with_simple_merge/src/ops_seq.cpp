#include "seq/yasakova_t_quick_sort_with_simple_merge/include/ops_seq.hpp"

#include <stack>

namespace yasakova_t_quick_sort_with_simple_merge_seq {

void performIterativeQuickSort(std::vector<int>& data) {
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

bool QuickSortWithMergeSeq::validation() {
  internal_order_test();

  return taskData->inputs_count[0] > 0;
}

bool QuickSortWithMergeSeq::pre_processing() {
  internal_order_test();

  auto* inputData = reinterpret_cast<int*>(taskData->inputs[0]);
  int inputSize = taskData->inputs_count[0];

  data_vector.assign(inputData, inputData + inputSize);

  return true;
}

bool QuickSortWithMergeSeq::run() {
  internal_order_test();

  performIterativeQuickSort(data_vector);

  return true;
}

bool QuickSortWithMergeSeq::post_processing() {
  internal_order_test();

  auto* outputData = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(data_vector.begin(), data_vector.end(), outputData);

  return true;
}

}  // namespace yasakova_t_quick_sort_with_simple_merge_seq