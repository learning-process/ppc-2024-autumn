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

  if (!taskData->outputs.empty() && !taskData->outputs_count.empty()) {
    return false;
  }

  return true;
}

bool TestTaskMPI::run() {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  size_t n = data_.size();
  std::vector<int> local_data;

  if (rank == 0) {
    size_t chunk_size = (n + size - 1) / size;
    for (int i = 1; i < size; ++i) {
      size_t start_idx = i * chunk_size;
      size_t end_idx = std::min(start_idx + chunk_size, n);
      MPI_Send(data_.data() + start_idx, static_cast<int>(end_idx - start_idx), MPI_INT, i, 0, MPI_COMM_WORLD);
    }
    local_data.assign(data_.begin(), data_.begin() + std::min(chunk_size, n));
  } else {
    MPI_Status status;
    size_t chunk_size = (n + size - 1) / size;
    size_t start_idx = rank * chunk_size;
    size_t end_idx = std::min(start_idx + chunk_size, n);

    local_data.resize(end_idx - start_idx);
    MPI_Recv(local_data.data(), static_cast<int>(end_idx - start_idx), MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
  }

  for (std::size_t gap = local_data.size() / 2; gap > 0; gap /= 2) {
    for (std::size_t i = gap; i < local_data.size(); ++i) {
      int temp = local_data[i];
      std::size_t j;
      for (j = i; j >= gap && local_data[j - gap] > temp; j -= gap) {
        local_data[j] = local_data[j - gap];
      }
      local_data[j] = temp;
    }
  }

  if (rank == 0) {
    std::copy(local_data.begin(), local_data.end(), data_.begin());
    for (int i = 1; i < size; ++i) {
      MPI_Status status;
      size_t chunk_size = (n + size - 1) / size;
      size_t start_idx = i * chunk_size;
      size_t end_idx = std::min(start_idx + chunk_size, n);

      MPI_Recv(data_.data() + start_idx, static_cast<int>(end_idx - start_idx), MPI_INT, i, 0, MPI_COMM_WORLD, &status);
    }
  } else {
    MPI_Send(local_data.data(), static_cast<int>(local_data.size()), MPI_INT, 0, 0, MPI_COMM_WORLD);
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
