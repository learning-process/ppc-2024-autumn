#include "seq/kozlova_e_radix_batcher_sort/include/ops_seq.hpp"

#include <cstring>

bool kozlova_e_radix_batcher_sort_seq::RadixSortSequential::pre_processing() {
  internal_order_test();
  input_size = taskData->inputs_count[0];
  data.resize(input_size);
  auto* mas = reinterpret_cast<double*>(taskData->inputs[0]);
  for (int i = 0; i < input_size; i++) data[i] = mas[i];
  return true;
}

bool kozlova_e_radix_batcher_sort_seq::RadixSortSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool kozlova_e_radix_batcher_sort_seq::RadixSortSequential::run() {
  internal_order_test();
  radixSort(data);
  return true;
}

bool kozlova_e_radix_batcher_sort_seq::RadixSortSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < data.size(); i++) reinterpret_cast<double*>(taskData->outputs[0])[i] = data[i];
  return true;
}

void kozlova_e_radix_batcher_sort_seq::RadixSortSequential::radixSort(std::vector<double>& a) {
  std::vector<uint64_t> bit_representation(a.size());

  for (size_t i = 0; i < a.size(); ++i) {
    uint64_t bits = *reinterpret_cast<uint64_t*>(&a[i]);
    if ((bits >> 63) != 0u) {
      bit_representation[i] = ~bits;
    } else {
      bit_representation[i] = bits | 0x8000000000000000;
    }
  }

  for (int bit = 0; bit < 64; ++bit) {
    std::vector<uint64_t> output(a.size());
    int count[2] = {0};

    for (size_t i = 0; i < bit_representation.size(); ++i) {
      count[(bit_representation[i] >> bit) & 1]++;
    }

    count[1] += count[0];

    for (int i = bit_representation.size() - 1; i >= 0; --i) {
      int val = (bit_representation[i] >> bit) & 1;
      output[--count[val]] = bit_representation[i];
    }

    bit_representation = output;
  }

  for (size_t i = 0; i < a.size(); ++i) {
    uint64_t bits = bit_representation[i];
    if ((bits & 0x8000000000000000) != 0u) {
      bits &= ~0x8000000000000000;
    } else {
      bits = ~bits;
    }
    memcpy(&a[i], &bits, sizeof(double));
  }
}
