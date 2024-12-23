#include "mpi/petrov_a_Shell_sort/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

namespace petrov_a_Shell_sort_mpi {

bool TestTaskMPI::validation() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (data_.empty()) {
    if (rank == 0) {
      std::cerr << "Input data is empty." << std::endl;
    }
    return false;
  }
  return true;
}

bool TestTaskMPI::pre_processing() {
  int world_size;
  int world_rank;

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

void odd_even_merge(std::vector<int>& data, int size) {
  for (int step = 1; step < size; step *= 2) {
    for (int i = step % 2; i < size - 1; i += 2) {
      if (data[i] > data[i + 1]) {
        std::swap(data[i], data[i + 1]);
      }
    }
  }
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

  int world_size;
  int world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::vector<int> merged_data;
  for (int step = 1; step < world_size; step *= 2) {
    if (world_rank % (2 * step) == 0) {
      if (world_rank + step < world_size) {
        int received_size;
        MPI_Status status;

        MPI_Recv(&received_size, 1, MPI_INT, world_rank + step, 0, MPI_COMM_WORLD, &status);
        std::vector<int> received_data(received_size);
        MPI_Recv(received_data.data(), received_size, MPI_INT, world_rank + step, 0, MPI_COMM_WORLD, &status);

        merged_data = local_data_;
        merged_data.insert(merged_data.end(), received_data.begin(), received_data.end());
        std::inplace_merge(merged_data.begin(), merged_data.begin() + local_data_.size(), merged_data.end());

        local_data_ = merged_data;
      }
    } else if (world_rank % step == 0) {
      int parent = world_rank - step;
      int local_size = static_cast<int>(local_data_.size());

      MPI_Send(&local_size, 1, MPI_INT, parent, 0, MPI_COMM_WORLD);
      MPI_Send(local_data_.data(), local_size, MPI_INT, parent, 0, MPI_COMM_WORLD);
      break;
    }
  }

  return true;
}

bool TestTaskMPI::post_processing() {
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) {
    data_ = local_data_;
  }

  return true;
}

}  // namespace petrov_a_Shell_sort_mpi
