#include "mpi/petrov_a_Shell_sort/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

namespace petrov_a_Shell_sort_mpi {

void shell_sort(std::vector<int>& arr) {
  size_t gap = arr.size() / 2;
  while (gap > 0) {
    for (size_t i = gap; i < arr.size(); ++i) {
      int temp = arr[i];
      size_t j = i;
      while (j >= gap && arr[j - gap] > temp) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = temp;
    }
    gap /= 2;
  }
}

bool TestTaskMPI::validation() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (data_.empty()) {
    if (rank == 0) {
      std::cerr << "Input data is empty. Validation failed." << std::endl;
    }
    return false;
  }

  if (rank == 0) {
    std::cerr << "Validation passed. Input data size: " << data_.size() << std::endl;
  }

  return true;
}

bool TestTaskMPI::pre_processing() {
  int world_size;
  int world_rank;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (data_.empty()) {
    if (world_rank == 0) {
      std::cerr << "Input data is empty; cannot proceed with sorting." << std::endl;
    }
    return false;
  }

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
  shell_sort(local_data_);
  return true;
}

bool TestTaskMPI::post_processing() {
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int vector_size = static_cast<int>(data_.size());
  if (world_rank == 0) {
    data_.resize(std::accumulate(send_counts_.begin(), send_counts_.end(), 0));
  }

  MPI_Gatherv(local_data_.data(), static_cast<int>(local_data_.size()), MPI_INT, data_.data(), send_counts_.data(),
              displs_.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    for (size_t i = 1; i < send_counts_.size(); ++i) {
      std::inplace_merge(data_.begin(), data_.begin() + displs_[i], data_.begin() + displs_[i] + send_counts_[i]);
    }
  }

  return true;
}

}  // namespace petrov_a_Shell_sort_mpi
