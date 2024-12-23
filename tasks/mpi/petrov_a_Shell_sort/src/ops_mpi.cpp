#include "mpi/petrov_a_Shell_sort/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

namespace petrov_a_Shell_sort_mpi {

bool TestTaskMPI::validation() {
  if (data_.empty()) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank == 0) {
      std::cerr << "Input data is empty." << std::endl;
    }
    return false;
  }
  return true;
}

bool TestTaskMPI::pre_processing() {
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int vector_size = static_cast<int>(data_.size());
  if (world_size > vector_size) {
    if (world_rank == 0) {
      std::cerr << "Number of processes exceeds the data size." << std::endl;
    }
    return false;
  }

  send_counts_.resize(world_size, vector_size / world_size);
  displs_.resize(world_size, 0);

  int remainder = vector_size % world_size;
  for (int i = 0; i < world_size; ++i) {
    if (i < remainder) {
      send_counts_[i]++;
    }
    if (i > 0) {
      displs_[i] = displs_[i - 1] + send_counts_[i - 1];
    }
  }

  local_data_.resize(send_counts_[world_rank]);
  MPI_Scatterv(data_.data(), send_counts_.data(), displs_.data(), MPI_INT, local_data_.data(), send_counts_[world_rank],
               MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool TestTaskMPI::run() {
  for (size_t gap = local_data_.size() / 2; gap > 0; gap /= 2) {
    for (size_t i = gap; i < local_data_.size(); ++i) {
      int temp = local_data_[i];
      size_t j = i;
      while (j >= gap && local_data_[j - gap] > temp) {
        local_data_[j] = local_data_[j - gap];
        j -= gap;
      }
      local_data_[j] = temp;
    }
  }
  return true;
}

bool TestTaskMPI::post_processing() {
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) {
    data_.resize(std::accumulate(send_counts_.begin(), send_counts_.end(), 0));
  }
  MPI_Gatherv(local_data_.data(), local_data_.size(), MPI_INT, data_.data(), send_counts_.data(), displs_.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    for (size_t i = 1; i < send_counts_.size(); ++i) {
      std::inplace_merge(data_.begin(), data_.begin() + displs_[i], data_.begin() + displs_[i] + send_counts_[i]);
    }
  }

  return true;
}

}  // namespace petrov_a_Shell_sort_mpi