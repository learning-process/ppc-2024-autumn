// Copyright 2023 Nesterov Alexander
#include "mpi/zaytsev_bitwise_sort_evenodd_Batcher/include/ops_mpi.hpp"

#include <algorithm>
#include <vector>

void zaytsev_bitwise_sort_evenodd_Batcher::radix_sort(std::vector<int>& data, int min_value, int max_value) {
  std::vector<int> buffer(data.size());

  int max_bits = 0;
  while (max_value > 0) {
    max_value >>= 1;
    ++max_bits;
  }

  for (int bit = 0; bit < max_bits; ++bit) {
    size_t zero_count = 0;

    for (int num : data) {
      if ((num & (1 << bit)) == 0) {
        buffer[zero_count++] = num;
      }
    }
    for (int num : data) {
      if ((num & (1 << bit)) != 0) {
        buffer[zero_count++] = num;
      }
    }

    data = buffer;
  }

  if (min_value < 0) {
    for (int& num : data) {
      num += min_value;
    }
  }
}

bool zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  int* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
  int input_count = taskData->inputs_count[0];
  data_.assign(input_data, input_data + input_count);

  return true;
}

bool zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskSequential::run() {
  internal_order_test();

  if (data_.empty()) {
    return true;
  }

  int min_value = *std::min_element(data_.begin(), data_.end());

  if (min_value < 0) {
    for (int& num : data_) {
      num -= min_value;
    }
  }

  int max_value = *std::max_element(data_.begin(), data_.end());

  radix_sort(data_, min_value, max_value);

  return true;
}

bool zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskSequential::post_processing() {
  internal_order_test();

  int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(data_.begin(), data_.end(), output_data);

  return true;
}

bool zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  }

  return true;
}

bool zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == taskData->inputs_count[0];
  }
  return true;
}

bool zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskParallel::run() {
  internal_order_test();

  int world_size = world.size();
  int world_rank = world.rank();

  int total_size = 0;
  if (world_rank == 0) {
    total_size = input_.size();
  }
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int chunk_size = total_size / world_size;
  int remainder = total_size % world_size;

  int local_size = chunk_size + (world_rank < remainder ? 1 : 0);
  std::vector<int> local_data(local_size);

  std::vector<int> counts(world_size);
  std::vector<int> displacements(world_size, 0);
  if (world_rank == 0) {
    for (int i = 0; i < world_size; ++i) {
      counts[i] = chunk_size + (i < remainder ? 1 : 0);
      if (i > 0) {
        displacements[i] = displacements[i - 1] + counts[i - 1];
      }
    }
  }

  MPI_Scatterv(input_.data(), counts.data(), displacements.data(), MPI_INT, local_data.data(), local_size, MPI_INT, 0,
               MPI_COMM_WORLD);

  int local_min = local_data.empty() ? 0 : *std::min_element(local_data.begin(), local_data.end());
  int global_min = 0;
  MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (global_min < 0) {
    for (int& num : local_data) {
      num -= global_min;
    }
  }
  int max_value = local_data.empty() ? 0 : *std::max_element(local_data.begin(), local_data.end());
  int global_max_value = 0;
  MPI_Allreduce(&max_value, &global_max_value, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  radix_sort(local_data, global_min, global_max_value);

  for (int step = 0; step < world_size; ++step) {
    int neighbor;
    if (step % 2 == 0) {
      neighbor = (world_rank % 2 == 0) ? world_rank + 1 : world_rank - 1;
    } else {
      neighbor = (world_rank % 2 == 0) ? world_rank - 1 : world_rank + 1;
    }

    if (neighbor >= 0 && neighbor < world_size) {
      std::vector<int> neighbor_data(local_size);
      MPI_Sendrecv(local_data.data(), local_size, MPI_INT, neighbor, 0, neighbor_data.data(), local_size, MPI_INT,
                   neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::vector<int> merged_data(local_data.size() + neighbor_data.size());
      std::merge(local_data.begin(), local_data.end(), neighbor_data.begin(), neighbor_data.end(), merged_data.begin());

      if (world_rank < neighbor) {
        local_data.assign(merged_data.begin(), merged_data.begin() + local_data.size());
      } else {
        local_data.assign(merged_data.end() - local_data.size(), merged_data.end());
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Gatherv(local_data.data(), local_data.size(), MPI_INT, input_.data(), counts.data(), displacements.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(input_.begin(), input_.end(), output_data);
  }

  return true;
}
