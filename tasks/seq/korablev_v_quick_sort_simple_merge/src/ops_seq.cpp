#include "seq/korablev_v_quick_sort_simple_merge/include/ops_seq.hpp"

#include <cmath>

std::vector<int> korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential::merge(
    const std::vector<int>& left, const std::vector<int>& right) {
  std::vector<int> result;
  result.reserve(left.size() + right.size());

  size_t i = 0;
  size_t j = 0;

  while (i < left.size() && j < right.size()) {
    if (left[i] < right[j]) {
      result.push_back(left[i++]);
    } else {
      result.push_back(right[j++]);
    }
  }

  result.insert(result.end(), left.begin() + i, left.end());
  result.insert(result.end(), right.begin() + j, right.end());

  return result;
}

std::vector<int> korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential::quick_sort_with_merge(
    const std::span<int>& arr) {
  if (arr.size() <= 1) {
    std::vector<int> res;
    res.assign(arr.begin(), arr.end());
    return res;
  }

  int pivot = arr[arr.size() / 2];
  std::vector<int> left;
  std::vector<int> right;
  std::vector<int> equal;

  for (const auto& elem : arr) {
    if (elem < pivot) {
      left.emplace_back(elem);
    } else if (elem > pivot) {
      right.emplace_back(elem);
    } else {
      equal.emplace_back(elem);
    }
  }

  std::vector<int> sortedLeft = quick_sort_with_merge(left);
  std::vector<int> sortedRight = quick_sort_with_merge(right);

  std::vector<int> merged = merge(sortedLeft, equal);
  return merge(merged, sortedRight);
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