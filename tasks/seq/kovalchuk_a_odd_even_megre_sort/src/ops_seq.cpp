#include "seq/kovalchuk_a_odd_even_megre_sort/include/ops_seq.hpp"

#include <algorithm>
#include <random>
#include <string>
#include <vector>

void batcher_merge(std::vector<int>& array, int start, int mid, int end) {
  int n = end - start;
  if (n <= 1) return;

  std::vector<int> even_array;
  std::vector<int> odd_array;
  for (int i = start; i < end; ++i) {
    if (i % 2 == start % 2) {
      even_array.push_back(array[i]);
    } else {
      odd_array.push_back(array[i]);
    }
  }

  batcher_merge(even_array, 0, even_array.size() / 2, even_array.size());
  batcher_merge(odd_array, 0, odd_array.size() / 2, odd_array.size());

  std::merge(even_array.begin(), even_array.end(), odd_array.begin(), odd_array.end(), array.begin() + start);
}

void batcher_sort(std::vector<int>& array, int start, int end) {
  if (end - start <= 1) return;

  int mid = (start + end) / 2;

  batcher_sort(array, start, mid);
  batcher_sort(array, mid, end);

  batcher_merge(array, start, mid, end);
}

bool kovalchuk_a_odd_even_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init array
  if (taskData->inputs_count[0] > 0) {
    array_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], array_.begin());
  } else {
    array_ = std::vector<int>();
  }
  // Init result vector
  result_ = std::vector<int>(taskData->inputs_count[0], 0);
  return true;
}

bool kovalchuk_a_odd_even_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool kovalchuk_a_odd_even_seq::TestTaskSequential::run() {
  internal_order_test();
  if (!array_.empty()) {
    batcher_sort(array_, 0, array_.size());
    result_ = array_;
  }
  return true;
}

bool kovalchuk_a_odd_even_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  std::copy(result_.begin(), result_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  return true;
}