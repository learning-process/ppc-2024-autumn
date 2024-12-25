#include "mpi/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_mpi.hpp"

void BitwiseCountingSort(std::vector<uint64_t>& keys, int shift) {
  std::vector<uint64_t> temp(keys.size());
  size_t count[256 + 1] = {0};

  for (size_t i = 0; i < keys.size(); ++i) {
    uint8_t byte = static_cast<uint8_t>((keys[i] >> shift) & ((1 << 8) - 1));
    ++count[byte + 1];
  }

  std::partial_sum(count, count + 256 + 1, count);

  for (size_t i = 0; i < keys.size(); ++i) {
    uint8_t byte = static_cast<uint8_t>((keys[i] >> shift) & ((1 << 8) - 1));
    temp[count[byte]++] = keys[i];
  }

  std::copy(temp.begin(), temp.end(), keys.begin());
}

void SortDoubleByBits(std::vector<double>& data) {
  std::vector<uint64_t> keys(data.size());

  for (size_t i = 0; i < data.size(); ++i) {
    uint64_t double_as_uint64;
    std::memcpy(&double_as_uint64, &data[i], sizeof(double));

    double_as_uint64 = (double_as_uint64 & (1ULL << 63)) ? ~double_as_uint64 : (double_as_uint64 | (1ULL << 63));
    keys[i] = double_as_uint64;
  }

  for (int shift = 0; shift < 64; shift += 8) {
    BitwiseCountingSort(keys, shift);
  }

  for (size_t i = 0; i < data.size(); ++i) {
    uint64_t double_as_uint64 = keys[i];

    double_as_uint64 = (double_as_uint64 & (1ULL << 63)) ? (double_as_uint64 & ~(1ULL << 63)) : ~double_as_uint64;
    std::memcpy(&data[i], &double_as_uint64, sizeof(double));
  }
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input.resize(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input.begin());
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs_count[0] <= 0) {
    return false;
  }
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  SortDoubleByBits(input);
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  auto* output = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(input.begin(), input.end(), output);
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input.resize(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input.begin());
    return true;
  }
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs_count[0] <= 0) {
      return false;
    }
    return true;
  }
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* output = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(input.begin(), input.end(), output);
    return true;
  }
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  const int rank = world.rank();
  const int total_processes = world.size();
  const int chunk_size = taskData->inputs_count[0] / total_processes;
  const int extra_elements = taskData->inputs_count[0] % total_processes;

  std::vector<int> counts(total_processes);
  std::vector<int> displacements(total_processes);

  if (rank == 0) {
    for (int i = 0; i < total_processes; ++i) {
      counts[i] = chunk_size + (i < extra_elements ? 1 : 0);
    }
    displacements[0] = 0;
    for (int i = 1; i < total_processes; ++i) {
      displacements[i] = displacements[i - 1] + counts[i - 1];
    }
  }

  MPI_Bcast(counts.data(), total_processes, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displacements.data(), total_processes, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<double> local_data(counts[rank]);
  MPI_Scatterv(rank == 0 ? input.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE, local_data.data(),
               counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  SortDoubleByBits(local_data);

  execute_merge(local_data, rank, total_processes);

  if (rank == 0) {
    res.swap(local_data);
  }

  return true;
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel::execute_merge(
    std::vector<double>& local_data, int rank, int total_processes) {
  int merge_steps = 0;
  int tmp_process_count = total_processes;
  while (tmp_process_count > 1) {
    tmp_process_count = (tmp_process_count + 1) / 2;
    ++merge_steps;
  }

  int current_group_size = 1;
  for (int step = 0; step < merge_steps; ++step) {
    const int partner_rank = rank + current_group_size;
    const bool can_merge = (rank % (current_group_size * 2) == 0);
    const bool has_partner = (partner_rank < total_processes);

    if (can_merge && has_partner) {
      int partner_data_size;
      MPI_Recv(&partner_data_size, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::vector<double> partner_data(partner_data_size);
      MPI_Recv(partner_data.data(), partner_data_size, MPI_DOUBLE, partner_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::vector<double> merged_result;
      merged_result.reserve(local_data.size() + partner_data.size());

      std::merge(local_data.begin(), local_data.end(), partner_data.begin(), partner_data.end(),
                 std::back_inserter(merged_result));

      local_data = std::move(merged_result);
    } else if (!can_merge && (rank % (current_group_size * 2) == current_group_size)) {
      int receiver_rank = rank - current_group_size;
      int local_data_size = static_cast<int>(local_data.size());

      MPI_Send(&local_data_size, 1, MPI_INT, receiver_rank, 0, MPI_COMM_WORLD);
      MPI_Send(local_data.data(), local_data_size, MPI_DOUBLE, receiver_rank, 1, MPI_COMM_WORLD);

      local_data.clear();
    }

    current_group_size *= 2;
  }
}