#include "mpi/petrov_a_Shell_sort/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <iostream>
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

bool validation(const std::vector<int>& data) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (data.empty()) {
    if (rank == 0) {
      std::cerr << "Input data is empty. Validation failed." << std::endl;
    }
    return false;
  }

  if (rank == 0) {
    std::cerr << "Validation passed. Input data size: " << data.size() << std::endl;
  }

  return true;
}

bool pre_processing(const std::vector<int>& data, std::vector<int>& local_data) {
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (data.empty()) {
    if (world_rank == 0) {
      std::cerr << "Input data is empty; cannot proceed with sorting." << std::endl;
    }
    return false;
  }

  int vector_size = static_cast<int>(data.size());
  if (world_size > vector_size) {
    if (world_rank == 0) {
      std::cerr << "Number of processes exceeds the data size." << std::endl;
    }
    return false;
  }

  std::vector<int> send_counts(world_size, vector_size / world_size);
  std::vector<int> displs(world_size, 0);

  int remainder = vector_size % world_size;
  for (int i = 0; i < world_size; ++i) {
    if (i < remainder) {
      send_counts[i]++;
    }
    if (i > 0) {
      displs[i] = displs[i - 1] + send_counts[i - 1];
    }
  }

  local_data.resize(send_counts[world_rank]);
  MPI_Scatterv(data.data(), send_counts.data(), displs.data(), MPI_INT, local_data.data(), send_counts[world_rank],
               MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool run(std::vector<int>& local_data) {
  shell_sort(local_data);
  return true;
}

bool post_processing(std::vector<int>& data, const std::vector<int>& local_data, const std::vector<int>& send_counts,
                     const std::vector<int>& displs) {
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int vector_size = std::accumulate(send_counts.begin(), send_counts.end(), 0);

  if (world_rank == 0) {
    data.resize(vector_size);
  }

  MPI_Gatherv(local_data.data(), static_cast<int>(local_data.size()), MPI_INT, data.data(), send_counts.data(),
              displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    for (size_t i = 1; i < send_counts.size(); ++i) {
      std::inplace_merge(data.begin(), data.begin() + displs[i], data.begin() + displs[i] + send_counts[i]);
    }
  }

  return true;
}

bool execute_task(std::vector<int>& data) {
  std::vector<int> local_data;
  if (!validation(data)) {
    return false;
  }

  if (!pre_processing(data, local_data)) {
    return false;
  }

  run(local_data);

  std::vector<int> send_counts(data.size(), 0);
  std::vector<int> displs(data.size(), 0);

  if (!post_processing(data, local_data, send_counts, displs)) {
    return false;
  }

  return true;
}

}  // namespace petrov_a_Shell_sort_mpi
