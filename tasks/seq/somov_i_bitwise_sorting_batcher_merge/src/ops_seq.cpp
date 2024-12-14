#include "seq/somov_i_bitwise_sorting_batcher_merge/include/ops_seq.hpp"

void countingSortByByte(std::vector<uint64_t> &data, std::vector<uint64_t> &output, int byteIndex) {
  const int base = 256;
  std::array<int, base> count = {0};
  for (uint64_t num : data) {
    int byteValue = (num >> (byteIndex * 8)) & 0xFF;
    count[byteValue]++;
  }
  for (int i = 1; i < base; i++) {
    count[i] += count[i - 1];
  }
  for (int i = data.size() - 1; i >= 0; i--) {
    int byteValue = (data[i] >> (byteIndex * 8)) & 0xFF;
    output[--count[byteValue]] = data[i];
  }
}

void radixSort(std::vector<uint64_t> &data) {
  const int bytesCount = 8;
  std::vector<uint64_t> temp(data.size());

  for (int byteIndex = 0; byteIndex < bytesCount; byteIndex++) {
    countingSortByByte(data, temp, byteIndex);
    data.swap(temp);
  }
}

void radix_sort_double(std::vector<double> &arr) {
  size_t n = arr.size();
  if (n == 0) return;

  std::vector<double> positive, negative;
  for (size_t i = 0; i < n; ++i) {
    if (arr[i] < 0) {
      negative.push_back(arr[i]);
    } else {
      positive.push_back(arr[i]);
    }
  }

  std::vector<uint64_t> posData(positive.size());
  for (size_t i = 0; i < positive.size(); ++i) {
    std::memcpy(&posData[i], &positive[i], sizeof(double));
  }

  radixSort(posData);

  for (size_t i = 0; i < positive.size(); ++i) {
    std::memcpy(&positive[i], &posData[i], sizeof(double));
  }

  std::vector<uint64_t> negData(negative.size());
  for (size_t i = 0; i < negative.size(); ++i) {
    std::memcpy(&negData[i], &negative[i], sizeof(double));
    negData[i] = ~negData[i];
  }

  radixSort(negData);

  for (size_t i = 0; i < negative.size(); ++i) {
    negData[i] = ~negData[i];
    std::memcpy(&negative[i], &negData[i], sizeof(double));
  }

  arr.clear();
  arr.insert(arr.end(), negative.begin(), negative.end());
  arr.insert(arr.end(), positive.begin(), positive.end());
}

bool somov_i_bitwise_sorting_batcher_merge_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_.resize(taskData->inputs_count[0]);
  auto *tmp_ptr = reinterpret_cast<double *>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  return true;
}

bool somov_i_bitwise_sorting_batcher_merge_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0;
}

bool somov_i_bitwise_sorting_batcher_merge_seq::TestTaskSequential::run() {
  internal_order_test();
  radix_sort_double(input_);
  res_ = input_;
  return true;
}

bool somov_i_bitwise_sorting_batcher_merge_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  auto *tmp_ptr = reinterpret_cast<double *>(taskData->outputs[0]);
  std::copy(res_.begin(), res_.end(), tmp_ptr);
  return true;
}
