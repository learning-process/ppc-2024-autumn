#include "seq/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_seq.hpp"

#include <cmath>
#include <queue>

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  int input_size = *(reinterpret_cast<int*>(taskData->inputs[0]));
  data.assign(reinterpret_cast<double*>(taskData->inputs[1]),
              reinterpret_cast<double*>(taskData->inputs[1]) + input_size);

  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential::validation() {
  internal_order_test();

  int input_size = *(reinterpret_cast<int*>(taskData->inputs[0]));

  return taskData->inputs_count[0] == 1 && taskData->inputs_count[1] == static_cast<size_t>(input_size) &&
         taskData->outputs_count[0] == static_cast<size_t>(input_size);
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential::run() {
  internal_order_test();

  std::vector<uint64_t> keys(data.size());
  convert_doubles_to_uint64(data, keys);
  radix_sort_uint64(keys);
  convert_uint64_to_doubles(keys, data);

  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  auto* out = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(data.begin(), data.end(), out);
  return true;
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential::convert_doubles_to_uint64(
    const std::vector<double>& data_, std::vector<uint64_t>& keys) {
  for (size_t i = 0; i < data_.size(); ++i) {
    uint64_t uint64_value;
    std::memcpy(&uint64_value, &data_[i], sizeof(double));

uint64_value = ((static_cast<bool>((uint64_value >> 63) & 1))) ? ~uint64_value : (uint64_value | (1ULL << 63));
    keys[i] = uint64_value;
  }
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential::convert_uint64_to_doubles(
    const std::vector<uint64_t>& keys, std::vector<double>& data_) {
  for (size_t i = 0; i < keys.size(); ++i) {
    uint64_t uint64_value = keys[i];

uint64_value = ((static_cast<bool>((uint64_value >> 63) & 1))) ? (uint64_value & ~(1ULL << 63)) : ~uint64_value;
    std::memcpy(&data_[i], &uint64_value, sizeof(double));
  }
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential::radix_sort_uint64(
    std::vector<uint64_t>& keys) {
  constexpr int BITS = 64;
  constexpr int RADIX = 256;
  std::vector<uint64_t> temp(keys.size());

  for (int shift = 0; shift < BITS; shift += 8) {
    size_t count[RADIX + 1] = {0};

    for (uint64_t key : keys) {
      ++count[((key >> shift) & 255) + 1];
    }

    for (int i = 0; i < RADIX; ++i) {
      count[i + 1] += count[i];
    }

    for (uint64_t key : keys) {
      temp[count[(key >> shift) & 255]++] = key;
    }

    keys.swap(temp);
  }
}
