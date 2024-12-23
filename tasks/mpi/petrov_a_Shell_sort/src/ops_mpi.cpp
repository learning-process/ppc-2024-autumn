#include "mpi/petrov_a_Shell_sort/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <stdexcept>
#include <vector>

namespace petrov_a_Shell_sort_mpi {

bool TestTaskMPI::validation() {
  return !data_.empty();
}

bool TestTaskMPI::pre_processing() {
  boost::mpi::communicator world;
  int world_size = world.size();
  int vector_size = static_cast<int>(data_.size());

  if (world_size > vector_size) {
    return false;
  }

  send_counts.resize(world_size);
  displacements.resize(world_size);

  int base_size = vector_size / world_size;
  int remainder = vector_size % world_size;

  for (int i = 0; i < world_size; ++i) {
    send_counts[i] = base_size + (i < remainder ? 1 : 0);
    displacements[i] = (i == 0) ? 0 : displacements[i - 1] + send_counts[i - 1];
  }

  local_data.resize(send_counts[world.rank()]);

  if (world.rank() == 0) {
    for (int i = 1; i < world_size; ++i) {
      MPI_Send(data_.data() + displacements[i], send_counts[i], MPI_INT, i, 0, MPI_COMM_WORLD);
    }
    std::copy(data_.begin(), data_.begin() + send_counts[0], local_data.begin());
  } else {
    MPI_Recv(local_data.data(), send_counts[world.rank()], MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  return true;
}

bool TestTaskMPI::run() {
  for (int gap = local_data.size() / 2; gap > 0; gap /= 2) {
    for (size_t i = gap; i < local_data.size(); ++i) {
      int temp = local_data[i];
      size_t j = i;
      while (j >= gap && local_data[j - gap] > temp) {
        local_data[j] = local_data[j - gap];
        j -= gap;
      }
      local_data[j] = temp;
    }
  }

  return true;
}

bool TestTaskMPI::post_processing() {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    data_.resize(send_counts[0]);
    std::copy(local_data.begin(), local_data.end(), data_.begin());
    for (int i = 1; i < world.size(); ++i) {
      std::vector<int> temp(send_counts[i]);
      MPI_Recv(temp.data(), send_counts[i], MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      data_.insert(data_.end(), temp.begin(), temp.end());
    }
    std::sort(data_.begin(), data_.end());
  } else {
    MPI_Send(local_data.data(), local_data.size(), MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

  return true;
}

}  // namespace petrov_a_Shell_sort_mpi
