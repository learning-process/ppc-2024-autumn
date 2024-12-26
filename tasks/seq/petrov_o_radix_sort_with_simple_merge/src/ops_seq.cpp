#include "seq/petrov_o_radix_sort_with_simple_merge/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace petrov_o_radix_sort_with_simple_merge_seq {

bool TestTaskSequential::validation() {
  internal_order_test();

  bool isValid = (!taskData->inputs_count.empty()) && (!taskData->inputs.empty()) && (!taskData->outputs.empty());

  return isValid;
}

bool TestTaskSequential::pre_processing() {
  internal_order_test();

  int size = taskData->inputs_count[0];
  input_.resize(size);

  int* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(input_data, input_data + size, input_.begin());

  res.resize(size);
  return true;
}

bool TestTaskSequential::run() {
  internal_order_test();

  for (auto& num : input_) {
    num ^= 0x80000000;
  }

  auto max_num = static_cast<unsigned int>(input_[0]);
  for (const auto& num : input_) {
    if (static_cast<unsigned int>(num) > max_num) {
      max_num = static_cast<unsigned int>(num);
    }
  }
  int num_bits = 0;
  const int MAX_BITS = sizeof(unsigned int) * 8;
  while (num_bits < MAX_BITS && (max_num >> num_bits) > 0) {
    num_bits++;
  }

  std::vector<int> output(input_.size());

  for (int bit = 0; bit < num_bits; ++bit) {
    int zero_count = 0;

    for (const auto& num : input_) {
      if (((static_cast<unsigned int>(num) >> bit) & 1) == 0) {
        zero_count++;
      }
    }

    int zero_index = 0;
    int one_index = zero_count;

    for (const auto& num : input_) {
      if (((static_cast<unsigned int>(num) >> bit) & 1) == 0) {
        output[zero_index++] = num;
      } else {
        output[one_index++] = num;
      }
    }

    input_ = output;
  }

  for (auto& num : input_) {
    num ^= 0x80000000;
  }

  res = input_;
  return true;
}

bool TestTaskSequential::post_processing() {
  internal_order_test();

  int* output_ = reinterpret_cast<int*>(taskData->outputs[0]);
  for (size_t i = 0; i < res.size(); ++i) {
    output_[i] = res[i];
  }
  std::copy(res.begin(), res.end(), output_);

  return true;
}

}  // namespace petrov_o_radix_sort_with_simple_merge_seq