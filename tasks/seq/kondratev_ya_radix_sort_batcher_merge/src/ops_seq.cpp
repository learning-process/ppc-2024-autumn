// Copyright 2024 Nesterov Alexander
#include "seq/kondratev_ya_radix_sort_batcher_merge/include/ops_seq.hpp"

void kondratev_ya_radix_sort_batcher_merge_seq::radixSortDouble(std::vector<double>& arr, int start, int end) {
  const int byte_size = 8;
  const int double_bit_size = byte_size * sizeof(double);
  const int full_byte_bit_mask = 0xFF;
  const int byte_range = std::pow(2, byte_size);
  const uint64_t sign_bit_mask = 1ULL << (double_bit_size - 1);

  int size = end - start + 1;
  std::vector<double> temp(size);

  // uint64_t because sizeof(uint64_t) == sizeof(double)
  auto* bits = reinterpret_cast<uint64_t*>(arr.data() + start);

  // Invert the sign bit for negative numbers
  for (int i = 0; i < size; i++) {
    if ((bool)(bits[i] >> (double_bit_size - 1))) {
      bits[i] = ~bits[i];
    } else {
      bits[i] |= sign_bit_mask;
    }
  }

  // Sorting by bits
  for (int shift = 0; shift < double_bit_size; shift += byte_size) {
    std::vector<int> count(byte_range, 0);
    for (int i = 0; i < size; i++) {
      count[(bits[i] >> shift) & full_byte_bit_mask]++;
    }

    std::partial_sum(count.begin(), count.end(), count.begin());

    for (int i = size - 1; i >= 0; i--) {
      int bucket = (bits[i] >> shift) & full_byte_bit_mask;
      temp[count[bucket] - 1] = arr[start + i];
      count[bucket]--;
    }
    std::copy(temp.begin(), temp.end(), arr.begin() + start);
  }

  // Restore inverted signs
  for (int i = 0; i < size; i++) {
    if (!(bool)(bits[i] & sign_bit_mask)) {
      bits[i] = ~bits[i];
    } else {
      bits[i] &= ~sign_bit_mask;
    }
  }
}

bool kondratev_ya_radix_sort_batcher_merge_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  auto* input = reinterpret_cast<double*>(taskData->inputs[0]);
  data_.assign(input, input + taskData->inputs_count[0]);

  return true;
}

bool kondratev_ya_radix_sort_batcher_merge_seq::TestTaskSequential::validation() {
  internal_order_test();

  return !taskData->inputs_count.empty() && taskData->inputs_count[0] > 0 && !taskData->outputs_count.empty() &&
         !taskData->outputs_count.empty() && taskData->outputs_count[0] == taskData->inputs_count[0] &&
         !taskData->inputs.empty() && taskData->inputs[0] != nullptr && !taskData->outputs.empty() &&
         taskData->outputs[0] != nullptr;
}

bool kondratev_ya_radix_sort_batcher_merge_seq::TestTaskSequential::run() {
  internal_order_test();

  radixSortDouble(data_, 0, data_.size() - 1);

  return true;
}

bool kondratev_ya_radix_sort_batcher_merge_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  auto* output = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(data_.begin(), data_.end(), output);

  return true;
}
