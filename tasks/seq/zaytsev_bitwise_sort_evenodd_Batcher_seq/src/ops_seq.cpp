// Copyright 2024 Nesterov Alexander
#include "seq/zaytsev_bitwise_sort_evenodd_Batcher_seq/include/ops_seq.hpp"

bool zaytsev_bitwise_sort_evenodd_Batcher_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  int* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
  int input_count = taskData->inputs_count[0];
  data_.assign(input_data, input_data + input_count);

  return true;
}

bool zaytsev_bitwise_sort_evenodd_Batcher_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool zaytsev_bitwise_sort_evenodd_Batcher_seq::TestTaskSequential::run() {
  internal_order_test();

  if (data_.empty()) {
    return true;
  }

  int min_value = *std::min_element(data_.begin(), data_.end());

  if (min_value < 0) {
    for (int& num : data_) {
      num -= min_value;
    }
  }

  int max_value = *std::max_element(data_.begin(), data_.end());
  int max_bits = 0;
  while (max_value > 0) {
    max_value >>= 1;
    ++max_bits;
  }

  std::vector<int> buffer(data_.size());

  for (int bit = 0; bit < max_bits; ++bit) {
    size_t zero_count = 0;

    for (int num : data_) {
      if ((num & (1 << bit)) == 0) {
        buffer[zero_count++] = num;
      }
    }

    for (int num : data_) {
      if ((num & (1 << bit)) != 0) {
        buffer[zero_count++] = num;
      }
    }

    data_ = buffer;
  }

  if (min_value < 0) {
    for (int& num : data_) {
      num += min_value;
    }
  }

  return true;
}

bool zaytsev_bitwise_sort_evenodd_Batcher_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(data_.begin(), data_.end(), output_data);

  return true;
}
