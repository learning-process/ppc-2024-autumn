#include "mpi/petrov_a_Shell_sort/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

namespace petrov_a_Shell_sort_mpi {

bool TestTaskMPI::pre_processing() {
  if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
    return false;
  }

  size_t input_size = taskData->inputs_count[0];
  const unsigned char* raw_data = reinterpret_cast<const unsigned char*>(taskData->inputs[0]);

  data_.resize(input_size);
  memcpy(data_.data(), raw_data, input_size * sizeof(int));

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
    size_t output_size = taskData->outputs_count[0];
    unsigned char* raw_output_data = reinterpret_cast<unsigned char*>(taskData->outputs[0]);

    memcpy(raw_output_data, data_.data(), output_size * sizeof(int));

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
