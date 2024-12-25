#include "seq/sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging/include/ops_seq.hpp"

bool sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  auto* input_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  input_data_.assign(input_ptr, input_ptr + taskData->inputs_count[0]);
  return true;
}

bool sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_seq::TestTaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs_count[0] == 0) {
    return true;
  }
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_seq::TestTaskSequential::run() {
  internal_order_test();
  radixSortWithSignHandling(input_data_);
  sorted_data_ = input_data_;
  return true;
}

bool sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  auto* output_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(sorted_data_.begin(), sorted_data_.end(), output_ptr);
  return true;
}

void sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_seq::radixSortWithSignHandling(
    std::vector<double>& data) {
  const int num_bits = sizeof(double) * 8;
  const int radix = 2;

  std::vector<double> positives;
  std::vector<double> negatives;

  for (double num : data) {
    if (num < 0) {
      negatives.push_back(-num);
    } else {
      positives.push_back(num);
    }
  }

  radixSort(positives, num_bits, radix);
  radixSort(negatives, num_bits, radix);

  for (double& num : negatives) {
    num = -num;
  }

  data.clear();
  data.insert(data.end(), negatives.rbegin(), negatives.rend());
  data.insert(data.end(), positives.begin(), positives.end());
}

void sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_seq::radixSort(std::vector<double>& data,
                                                                                     int num_bits, int radix) {
  std::vector<std::vector<double>> buckets(radix);
  std::vector<double> output(data.size());

  for (int exp = 0; exp < num_bits; ++exp) {
    for (auto& num : data) {
      uint64_t bits = *reinterpret_cast<uint64_t*>(&num);
      int digit = (bits >> exp) & 1;
      buckets[digit].push_back(num);
    }

    int index = 0;
    for (int i = 0; i < radix; ++i) {
      for (auto& num : buckets[i]) {
        output[index++] = num;
      }
      buckets[i].clear();
    }

    data = output;
  }
}
