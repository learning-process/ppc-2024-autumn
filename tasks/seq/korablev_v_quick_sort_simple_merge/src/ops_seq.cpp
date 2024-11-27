#include "seq/korablev_v_quick_sort_simple_merge/include/ops_seq.hpp"

#include <cmath>

std::vector<int> korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential::merge(
    std::vector<int>& left, std::vector<int>& right) {
  std::vector<int> result;
  size_t i = 0;
  size_t j = 0;

  while (i < left.size() && j < right.size()) {
    if (left[i] < right[j]) {
      result.emplace_back(left[i++]);
    } else {
      result.emplace_back(right[j++]);
    }
  }

  while (i < left.size()) {
    result.emplace_back(left[i++]);
  }
  while (j < right.size()) {
    result.emplace_back(right[j++]);
  }

  return result;
}

std::vector<int> korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential::quick_sort_with_merge(
    std::vector<int>& arr) {
  if (arr.size() <= 1) {
    std::vector<int> answ(arr.size());
    std::copy(arr.begin(), arr.end(), answ.begin());
    return answ;
  }

  int pivot = arr[arr.size() / 2];
  std::vector<int> left;
  std::vector<int> right;

  for (const auto& elem : arr) {
    if (elem < pivot) {
      left.emplace_back(elem);
    } else if (elem > pivot) {
      right.emplace_back(elem);
    }
  }

  std::vector<int> sortedLeft = quick_sort_with_merge(left);
  std::vector<int> sortedRight = quick_sort_with_merge(right);

  std::vector<int> merged = sortedLeft;
  for (const auto& elem : arr) {
    if (elem == pivot) {
      merged.push_back(elem);
    }
  }
  std::vector<int> finalResult = merge(merged, sortedRight);

  return finalResult;
}

bool korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential::pre_processing() {
  internal_order_test();

  size_t n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  auto* input_data = reinterpret_cast<int*>(taskData->inputs[1]);

  input_.assign(input_data, input_data + n);

  return true;
}

bool korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() != 2 || taskData->outputs_count.size() != 1) {
    return false;
  }

  size_t n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  if (n < 0 || taskData->inputs_count[1] != n) {
    return false;
  }

  if (taskData->outputs_count[0] != n) {
    return false;
  }

  return true;
}

bool korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential::run() {
  internal_order_test();

  output_ = quick_sort_with_merge(input_);

  return true;
}

bool korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < output_.size(); ++i) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = output_[i];
  }

  return true;
}