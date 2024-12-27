#include "seq/belov_a_radix_sort_with_batcher_mergesort/include/ops_seq.hpp"

using namespace std;

namespace belov_a_radix_batcher_mergesort_seq {
int RadixBatcherMergesortSequential::get_number_digit_capacity(bigint num) {
  return (num == 0 ? 1 : static_cast<int>(log10(abs(num))) + 1);
}

void RadixBatcherMergesortSequential::sort(vector<bigint>& arr) {
  vector<bigint> pos;
  vector<bigint> neg;

  for (const auto& num : arr) {
    (num >= 0 ? pos : neg).push_back(abs(num));
  }

  radix_sort(pos, false);
  radix_sort(neg, true);

  arr.clear();
  arr.reserve(neg.size() + pos.size());
  for (const auto& num : neg) arr.push_back(-num);
  arr.insert(arr.end(), pos.begin(), pos.end());
}

void RadixBatcherMergesortSequential::radix_sort(vector<bigint>& arr, bool invert) {
  if (arr.empty()) return;

  bigint max_val = *std::max_element(arr.begin(), arr.end());
  int max_val_digit_capacity = get_number_digit_capacity(max_val);
  int iter = 1;

  for (bigint digit_place = 1; iter <= max_val_digit_capacity; digit_place *= 10, ++iter) {
    counting_sort(arr, digit_place);
  }

  if (invert) std::reverse(arr.begin(), arr.end());
}

void RadixBatcherMergesortSequential::counting_sort(vector<bigint>& arr, bigint digit_place) {
  vector<bigint> output(arr.size());
  int count[10] = {};

  for (const auto& num : arr) {
    bigint index = (num / digit_place) % 10;
    count[index]++;
  }

  for (int i = 1; i < 10; i++) {
    count[i] += count[i - 1];
  }

  for (int i = arr.size() - 1; i >= 0; i--) {
    bigint num = arr[i];
    bigint index = (num / digit_place) % 10;
    output[--count[index]] = num;
  }

  std::copy(output.begin(), output.end(), arr.begin());
}

bool RadixBatcherMergesortSequential::pre_processing() {
  internal_order_test();

  n = taskData->inputs_count[0];
  auto* input_array_data = reinterpret_cast<bigint*>(taskData->inputs[0]);
  array.assign(input_array_data, input_array_data + n);

  return true;
}

bool RadixBatcherMergesortSequential::validation() {
  internal_order_test();

  return (taskData->inputs.size() == 1 && !(taskData->inputs_count.size() < 2) && taskData->inputs_count[0] != 0 &&
          (taskData->inputs_count[0] == taskData->inputs_count[1]) && !taskData->outputs.empty());
}

bool RadixBatcherMergesortSequential::run() {
  internal_order_test();

  sort(array);
  return true;
}

bool RadixBatcherMergesortSequential::post_processing() {
  internal_order_test();

  copy(array.begin(), array.end(), reinterpret_cast<bigint*>(taskData->outputs[0]));
  return true;
}

}  // namespace belov_a_radix_batcher_mergesort_seq
