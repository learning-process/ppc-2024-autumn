#include "mpi/petrov_a_Shell_sort/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

namespace petrov_a_Shell_sort_mpi {

bool TestTaskMPI::pre_processing() {
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_rank == 0) {
    size_t input_size = taskData->inputs_count[0];
    const auto* raw_data = reinterpret_cast<const unsigned char*>(taskData->inputs[0]);

    data_.resize(input_size);
    memcpy(data_.data(), raw_data, input_size * sizeof(int));
  }

  return true;
}

bool TestTaskMPI::validation() {
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_rank == 0) {
    if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
      return false;
    }

    if (taskData->outputs.empty() || taskData->outputs_count.empty()) {
      return false;
    }

    if (!taskData->outputs.empty() && taskData->outputs_count[0] == 0) {
      return false;
    }
  }
  return true;
}

bool TestTaskMPI::run() {
  int world_size;
  int world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int n;

  if (world_rank == 0) {
    n = data_.size();
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int local_size = n / world_size;
  std::vector<int> local_data(local_size);

  MPI_Scatter(data_.data(), local_size, MPI_INT, local_data.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

  for (int gap = local_size / 2; gap > 0; gap /= 2) {
    for (int i = gap; i < local_size; ++i) {
      int temp = local_data[i];
      int j;
      for (j = i; j >= gap && local_data[j - gap] > temp; j -= gap) {
        local_data[j] = local_data[j - gap];
      }
      local_data[j] = temp;
    }
  }

  MPI_Gather(local_data.data(), local_size, MPI_INT, data_.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    std::sort(data_.begin(), data_.end());
  }

  return true;
}

bool TestTaskMPI::post_processing() {
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) {
    size_t output_size = taskData->outputs_count[0];
    auto* raw_output_data = reinterpret_cast<unsigned char*>(taskData->outputs[0]);
    memcpy(raw_output_data, data_.data(), output_size * sizeof(int));
  }

  return true;
}

}  // namespace petrov_a_Shell_sort_mpi
