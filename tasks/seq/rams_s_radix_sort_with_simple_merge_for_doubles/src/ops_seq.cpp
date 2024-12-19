#include "seq/rams_s_radix_sort_with_simple_merge_for_doubles/include/ops_seq.hpp"

#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>

bool rams_s_radix_sort_with_simple_merge_for_doubles_seq::TaskSequential::pre_processing() {
  internal_order_test();

  auto *input_data = reinterpret_cast<double *>(taskData->inputs[0]);
  input = std::vector<double>(input_data, input_data + taskData->inputs_count[0]);
  result = std::vector<double>(taskData->outputs_count[0]);
  return true;
}

bool rams_s_radix_sort_with_simple_merge_for_doubles_seq::TaskSequential::validation() {
  internal_order_test();

  return taskData->inputs_count[0] >= 0 && taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool rams_s_radix_sort_with_simple_merge_for_doubles_seq::TaskSequential::run() {
  internal_order_test();

  const size_t radix = 8;
  const size_t histogram_size = 1 << radix;
  const size_t bits_per_item = sizeof(double) * CHAR_BIT;
  const size_t histograms_count = bits_per_item / radix;
  const size_t histogram_mask = histogram_size - 1;
  auto histograms = std::vector(histograms_count, std::vector<size_t>(histogram_size, 0));

  auto get_histogram_value = [&](size_t histogram_index, double item) -> auto & {
    const auto double_internal = std::bit_cast<uint64_t>(item);
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 4146)
#endif
    return histograms[histogram_index][((double_internal ^ (-(double_internal >> (bits_per_item - 1)) |
                                                            (static_cast<size_t>(1) << (bits_per_item - 1)))) >>
                                        (radix * histogram_index)) &
                                       histogram_mask];
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
  };

  for (const auto item : input) {
    for (size_t i = 0; i < histograms_count; i++) {
      get_histogram_value(i, item)++;
    }
  }

  for (auto &histogram : histograms) {
    size_t sum = 0;
    for (size_t i = 0; i < histogram_size; i++) {
      size_t count = histogram[i];
      histogram[i] = sum;
      sum += count;
    }
  }
  for (size_t i = 0; i < histograms_count; i++) {
    for (const auto item : input) {
      size_t &dest = get_histogram_value(i, item);
      result[dest] = item;
      dest++;
    }
    std::swap(input, result);
  }
  if constexpr (histograms_count % 2 == 0) {
    std::swap(input, result);
  }

  return true;
}

bool rams_s_radix_sort_with_simple_merge_for_doubles_seq::TaskSequential::post_processing() {
  internal_order_test();

  std::copy(result.begin(), result.end(), reinterpret_cast<double *>(taskData->outputs[0]));
  return true;
}
