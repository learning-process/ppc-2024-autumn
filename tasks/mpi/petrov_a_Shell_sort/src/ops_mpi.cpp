#include "mpi/petrov_a_Shell_sort/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <vector>

namespace petrov_a_Shell_sort_mpi {

bool TestTaskMPI::pre_processing() {
  if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
    return false;
  }

  const std::vector<int>& input_data = taskData->inputs[0];
  size_t input_size = taskData->inputs_count[0];
  data_ = input_data;

  return !data_.empty();
}

bool TestTaskMPI::validation() { return true; }

bool TestTaskMPI::run() {
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

bool TestTaskMPI::post_processing() {
  if (!taskData->outputs.empty() && !taskData->outputs_count.empty()) {
    std::vector<int>& output_data = taskData->outputs[0];
    size_t output_size = taskData->outputs_count[0];
    std::copy(data_.begin(), data_.begin() + std::min(output_size, data_.size()), output_data.begin());
  } else {
    return false;
  }
  return true;
}

void TestTaskMPI::compareAndSwap(int i, int j) {
  if (data_[i] > data_[j]) {
    std::swap(data_[i], data_[j]);
  }
}

}  // namespace petrov_a_Shell_sort_mpi
