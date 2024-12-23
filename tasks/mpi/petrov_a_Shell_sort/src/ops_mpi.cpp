#include "mpi/petrov_a_Shell_sort/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <stdexcept>
#include <vector>

namespace petrov_a_Shell_sort_mpi {

bool TestTaskMPI::validation() { return !data_.empty(); }

bool TestTaskMPI::pre_processing() {
  boost::mpi::communicator world;
  int world_size = world.size();
  int world_rank = world.rank();
  int vector_size = static_cast<int>(data_.size());

  if (world_size > vector_size) {
    return false;
  }

  int base_size = vector_size / world_size;
  int remainder = vector_size % world_size;

  send_counts_.resize(world_size);
  displacements_.resize(world_size);

  for (int i = 0; i < world_size; ++i) {
    send_counts_[i] = base_size + (i < remainder ? 1 : 0);
    displacements_[i] = (i > 0) ? (displacements_[i - 1] + send_counts_[i - 1]) : 0;
  }

  local_data_.resize(send_counts_[world_rank]);

  if (world_rank == 0) {
    for (int i = 1; i < world_size; ++i) {
      world.send(i, 0, &data_[displacements_[i]], send_counts_[i]);
    }
    std::copy(data_.begin(), data_.begin() + send_counts_[0], local_data_.begin());
  } else {
    world.recv(0, 0, local_data_.data(), send_counts_[world_rank]);
  }

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
  boost::mpi::communicator world;
  int world_rank = world.rank();
  int world_size = world.size();

  // Ручной сбор данных
  if (world_rank == 0) {
    std::copy(local_data_.begin(), local_data_.end(), data_.begin());
    for (int i = 1; i < world_size; ++i) {
      world.recv(i, 1, &data_[displacements_[i]], send_counts_[i]);
    }
  } else {
    world.send(0, 1, local_data_.data(), send_counts_[world_rank]);
  }

  return true;
}

}  // namespace petrov_a_Shell_sort_mpi
