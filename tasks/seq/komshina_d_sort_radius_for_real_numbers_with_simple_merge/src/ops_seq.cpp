#include "seq/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_seq.hpp"

#include <bitset>
#include <numeric>

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  input.resize(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input.begin());
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential::run() {
  internal_order_test();
  SortDoubleByBits(input);
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  auto* output = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(input.begin(), input.end(), output);
  return true;
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential::SortDoubleByBits(
    std::vector<double>& data) {
  std::vector<uint64_t> keys(data.size());

  for (size_t i = 0; i < data.size(); ++i) {
    uint64_t double_as_uint64;
    std::memcpy(&double_as_uint64, &data[i], sizeof(double));

    double_as_uint64 = ((double_as_uint64 & (1ULL << 63)) != 0) ? ~double_as_uint64 : (double_as_uint64 | (1ULL << 63));
    keys[i] = double_as_uint64;
  }

  for (int shift = 0; shift < 64; shift += 8) {
    BitwiseCountingSort(keys, shift);
  }

  for (size_t i = 0; i < data.size(); ++i) {
    uint64_t double_as_uint64 = keys[i];

    double_as_uint64 =
        ((double_as_uint64 & (1ULL << 63)) != 0) ? (double_as_uint64 & ~(1ULL << 63)) : ~double_as_uint64;
    std::memcpy(&data[i], &double_as_uint64, sizeof(double));
  }
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential::BitwiseCountingSort(
    std::vector<uint64_t>& keys, int shift) {
  std::vector<uint64_t> temp(keys.size());
  size_t count[256 + 1] = {0};

  for (size_t i = 0; i < keys.size(); ++i) {
    auto byte = static_cast<uint8_t>((keys[i] >> shift) & ((1 << 8) - 1));
    ++count[byte + 1];
  }

  std::partial_sum(count, count + 256 + 1, count);

  for (size_t i = 0; i < keys.size(); ++i) {
    auto byte = static_cast<uint8_t>((keys[i] >> shift) & ((1 << 8) - 1));
    temp[count[byte]++] = keys[i];
  }

  std::copy(temp.begin(), temp.end(), keys.begin());
}