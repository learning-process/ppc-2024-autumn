#include "seq/korablev_v_quick_sort_simple_merge/include/ops_seq.hpp"

#include <cmath>

std::vector<double> korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential::merge(
    std::vector<double>& left, std::vector<double>& right) {
  std::vector<double> result;
  size_t i = 0;
  size_t j = 0;

  while (i < left.size() && j < right.size()) {
    if (left[i] < right[j]) {
      result.push_back(left[i++]);
    } else {
      result.push_back(right[j++]);
    }
  }

  while (i < left.size()) {
    result.push_back(left[i++]);
  }
  while (j < right.size()) {
    result.push_back(right[j++]);
  }

  return result;
}

std::vector<double> korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential::quick_sort_with_merge(
    std::vector<double>& arr) {
  if (arr.size() <= 1) {
    return std::vector<double>(arr);
  }

  double pivot = arr[arr.size() / 2];
  std::vector<double> left;
  std::vector<double> right;

  for (const auto& elem : arr) {
    if (elem < pivot) {
      left.push_back(elem);
    } else if (elem > pivot) {
      right.push_back(elem);
    }
  }

  std::vector<double> sortedLeft = quick_sort_with_merge(left);
  std::vector<double> sortedRight = quick_sort_with_merge(right);

  std::vector<double> merged = sortedLeft;
  for (const auto& elem : arr) {
    if (elem == pivot) {
      merged.push_back(elem);
    }
  }
  std::vector<double> finalResult = merge(merged, sortedRight);

  return finalResult;
}

bool korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential::pre_processing() {
  internal_order_test();

  size_t n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  auto* input_data = reinterpret_cast<double*>(taskData->inputs[1]);

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
    reinterpret_cast<double*>(taskData->outputs[0])[i] = output_[i];
  }

  return true;
}