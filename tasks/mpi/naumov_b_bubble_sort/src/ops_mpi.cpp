#include "mpi/naumov_b_bubble_sort/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <vector>

bool naumov_b_bubble_sort_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  auto* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
  int input_size = taskData->inputs_count[0];
  input_.resize(input_size);
  std::copy(input_data, input_data + input_size, input_.begin());

  return true;
}

bool naumov_b_bubble_sort_seq::TestTaskSequential::validation() {
  internal_order_test();

  if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
    return false;
  }
  if (taskData->inputs_count[0] <= 0) {
    return false;
  }
  if (taskData->outputs_count.empty() || taskData->outputs_count[0] != taskData->inputs_count[0]) {
    return false;
  }

  return true;
}

bool naumov_b_bubble_sort_seq::TestTaskSequential::run() {
  internal_order_test();

  for (size_t i = 0; i < input_.size(); ++i) {
    for (size_t j = 0; j < input_.size() - i - 1; ++j) {
      if (input_[j] > input_[j + 1]) {
        std::swap(input_[j], input_[j + 1]);
      }
    }
  }

  return true;
}

bool naumov_b_bubble_sort_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  if (taskData->outputs.empty() || taskData->outputs_count[0] != input_.size()) {
    return false;
  }

  auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(input_.begin(), input_.end(), output_data);

  return true;
}

bool naumov_b_bubble_sort_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    auto* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
    int input_size = taskData->inputs_count[0];

    input_.resize(input_size);
    std::copy(input_data, input_data + input_size, input_.begin());
  }

  return true;
}

bool naumov_b_bubble_sort_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
      return false;
    }
    if (taskData->inputs_count[0] <= 0) {
      return false;
    }
    if (taskData->outputs_count.empty() || taskData->outputs_count[0] != taskData->inputs_count[0]) {
      return false;
    }
  }

  return true;
}

bool naumov_b_bubble_sort_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    total_size_ = input_.size();
  }
  MPI_Bcast(&total_size_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int base_chunk = total_size_ / size;
  int remainder = total_size_ % size;

  int local_size = base_chunk + (rank < remainder ? 1 : 0);
  local_input_.resize(local_size);

  if (rank == 0) {
    for (int i = 1; i < size; ++i) {
      int start_idx = i * base_chunk + std::min(i, remainder);
      int chunk_size = base_chunk + (i < remainder ? 1 : 0);
      MPI_Send(input_.data() + start_idx, chunk_size, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + local_size);
  } else {
    MPI_Recv(local_input_.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  for (size_t i = 0; i < local_input_.size(); ++i) {
    for (size_t j = 0; j < local_input_.size() - i - 1; ++j) {
      if (local_input_[j] > local_input_[j + 1]) {
        std::swap(local_input_[j], local_input_[j + 1]);
      }
    }
  }

  return true;
}

bool naumov_b_bubble_sort_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<int> gathered_data;

  if (rank == 0) {
    gathered_data.resize(total_size_);
    std::copy(local_input_.begin(), local_input_.end(), gathered_data.begin());

    for (int i = 1; i < size; ++i) {
      int start_idx = i * (total_size_ / size) + std::min(i, total_size_ % size);
      int chunk_size = total_size_ / size + (i < total_size_ % size ? 1 : 0);
      MPI_Recv(gathered_data.data() + start_idx, chunk_size, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    for (size_t i = 0; i < gathered_data.size(); ++i) {
      for (size_t j = 0; j < gathered_data.size() - i - 1; ++j) {
        if (gathered_data[j] > gathered_data[j + 1]) {
          std::swap(gathered_data[j], gathered_data[j + 1]);
        }
      }
    }

    auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(gathered_data.begin(), gathered_data.end(), output_data);
  } else {
    MPI_Send(local_input_.data(), local_input_.size(), MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

  return true;
}
