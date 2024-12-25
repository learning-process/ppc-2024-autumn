#include "seq/sarafanov_m_quick_sort_batcher_merge/include/ops_seq.hpp"

#include <algorithm>

namespace sarafanov_m_quick_sort_batcher_merge_seq {

void batcher_merge(std::vector<int>& arr, int left, int right) {
  int n = right - left + 1;
  int step = 1;

  while (step < n) {
    int k = step;
    while (k < n) {
      for (int i = left; i + k <= right; i++) {
        if (arr[i] > arr[i + k]) {
          std::swap(arr[i], arr[i + k]);
        }
      }
      k *= 2;
    }
    step *= 2;
  }
}

void quick_sort(std::vector<int>& arr, int left, int right) {
  if (right - left + 1 <= THRESHOLD) {
    std::sort(arr.begin() + left, arr.begin() + right + 1);
    return;
  }

  int pivot = arr[left + (right - left) / 2];
  int i = left;
  int j = right;

  while (i <= j) {
    while (arr[i] < pivot) i++;
    while (arr[j] > pivot) j--;
    if (i <= j) {
      std::swap(arr[i], arr[j]);
      i++;
      j--;
    }
  }

  if (left < j) quick_sort(arr, left, j);
  if (i < right) quick_sort(arr, i, right);
}

void quick_sort_with_batcher_merge(std::vector<int>& arr) {
  int size = arr.size();
  quick_sort(arr, 0, size - 1);

  batcher_merge(arr, 0, size - 1);
}

bool QuicksortBatcherMerge::validation() {
  internal_order_test();

  int val_arr_size = taskData->inputs_count[0];
  int val_out_arr_size = taskData->outputs_count[0];

  return val_arr_size > 0 && val_out_arr_size == val_arr_size && (val_arr_size & (val_arr_size - 1)) == 0;
}

bool QuicksortBatcherMerge::pre_processing() {
  internal_order_test();

  auto* vec_data = reinterpret_cast<int*>(taskData->inputs[0]);
  int vec_size = taskData->inputs_count[0];
  arr.assign(vec_data, vec_data + vec_size);

  return true;
}

bool QuicksortBatcherMerge::run() {
  internal_order_test();

  quick_sort_with_batcher_merge(arr);

  return true;
}

bool QuicksortBatcherMerge::post_processing() {
  internal_order_test();

  auto* out_vector = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(arr.begin(), arr.end(), out_vector);

  return true;
}

}  // namespace sarafanov_m_quick_sort_batcher_merge_seq
