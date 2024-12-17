#include "seq/petrov_a_Shell_sort/include/ops_seq.hpp"

#include <algorithm>
#include <iostream>
#include <limits>

namespace petrov_a_Shell_sort_seq {

bool TestTaskSequential::pre_processing() {
  if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
    return false;
  }

  const int* input_data = reinterpret_cast<const int*>(taskData->inputs[0]);
  size_t input_size = taskData->inputs_count[0];
  data_ = std::vector<int>(input_data, input_data + input_size);

  return !data_.empty
}

bool TestTaskSequential::validation() { return true; }

bool TestTaskSequential::run() {
  int n = data_.size();

  for (int gap = n / 2; gap > 0; gap /= 2) {
    for (int i = gap; i < n; ++i) {
      int temp = data_[i];
      int j;
      for (j = i; j >= gap && data_[j - gap] > temp; j -= gap) {
        data_[j] = data_[j - gap];
      }
      data_[j] = temp;
    }
  }

  return true;
}

bool TestTaskSequential::post_processing() {
  if (!taskData->outputs.empty() && !taskData->outputs_count.empty()) {
    int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
    size_t output_size = taskData->outputs_count[0];
    std::copy(data_.begin(), data_.begin() + std::min(output_size, data_.size()), output_data);
  } else {
    return false;
  }
  return true;
}

void TestTaskSequential::compareAndSwap(int i, int j) {
  if (data_[i] > data_[j]) {
    std::swap(data_[i], data_[j]);
  }
}

}  // namespace petrov_a_Shell_sort_seq
