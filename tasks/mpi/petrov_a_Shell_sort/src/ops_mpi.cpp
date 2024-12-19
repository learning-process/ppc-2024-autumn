#include "mpi/petrov_a_Shell_sort/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

namespace petrov_a_Shell_sort_mpi {

bool TestTaskMPI::pre_processing() {
  size_t input_size = taskData->inputs_count[0];
  const auto* raw_data = reinterpret_cast<const unsigned char*>(taskData->inputs[0]);

  data_.resize(input_size);
  memcpy(data_.data(), raw_data, input_size * sizeof(int));

  return true;
}

bool TestTaskMPI::validation() {
  if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
    return false;
  }
  if (taskData->outputs.empty() || taskData->outputs_count.empty()) {
    return false;
  }
  return true;
}

bool TestTaskMPI::run() {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  int size;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int n = data_.size();
  int local_n = n / size;
  int remainder = n % size;
  std::vector<int> local_data(local_n);

  MPI_Scatter(data_.data(), local_n, MPI_INT, local_data.data(), local_n, MPI_INT, 0, comm);

  for (int gap = local_n / 2; gap > 0; gap /= 2) {
    for (int i = gap; i < local_n; ++i) {
      int temp = local_data[i];
      int j;
      for (j = i; j >= gap && local_data[j - gap] > temp; j -= gap) {
        local_data[j] = local_data[j - gap];
      }
      local_data[j] = temp;
    }
  }

  MPI_Gather(local_data.data(), local_n, MPI_INT, data_.data(), local_n, MPI_INT, 0, comm);

  if (rank == 0 && remainder > 0) {
    std::vector<int> remainder_data(remainder);
    std::copy(data_.begin() + local_n * size, data_.end(), remainder_data.begin());

    for (int gap = remainder / 2; gap > 0; gap /= 2) {
      for (int i = gap; i < remainder; ++i) {
        int temp = remainder_data[i];
        int j;
        for (j = i; j >= gap && remainder_data[j - gap] > temp; j -= gap) {
          remainder_data[j] = remainder_data[j - gap];
        }
        remainder_data[j] = temp;
      }
    }

    std::vector<int> merged_data(local_n * size + remainder);
    std::merge(data_.begin(), data_.begin() + local_n * size, remainder_data.begin(), remainder_data.end(),
               merged_data.begin());

    data_ = merged_data;
  }

  if (rank == 0) {
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
  }

  return true;
}

bool TestTaskMPI::post_processing() {
  size_t output_size = taskData->outputs_count[0];
  auto* raw_output_data = reinterpret_cast<unsigned char*>(taskData->outputs[0]);
  memcpy(raw_output_data, data_.data(), output_size * sizeof(int));
  return true;
}

}  // namespace petrov_a_Shell_sort_mpi
