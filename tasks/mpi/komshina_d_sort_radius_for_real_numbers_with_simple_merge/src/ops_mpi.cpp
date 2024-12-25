#include "mpi/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_mpi.hpp"

#include <boost/mpi.hpp>
#include <cmath>
#include <queue>

namespace mpi = boost::mpi;

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskSequential::pre_processing() {
  internal_order_test();

  int input_size = *(reinterpret_cast<int*>(taskData->inputs[0]));
  data.assign(reinterpret_cast<double*>(taskData->inputs[1]),
              reinterpret_cast<double*>(taskData->inputs[1]) + input_size);

  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskSequential::validation() {
  internal_order_test();

  int input_size = *(reinterpret_cast<int*>(taskData->inputs[0]));

  return taskData->inputs_count[0] == 1 && taskData->inputs_count[1] == static_cast<size_t>(input_size) &&
         taskData->outputs_count[0] == static_cast<size_t>(input_size);
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskSequential::run() {
  internal_order_test();

  std::vector<uint64_t> keys(data.size());
  convert_doubles_to_uint64(data, keys);
  radix_sort_uint64(keys);
  convert_uint64_to_doubles(keys, data);

  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskSequential::post_processing() {
  internal_order_test();

  auto* out = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(data.begin(), data.end(), out);

  return true;
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskSequential::convert_doubles_to_uint64(
    const std::vector<double>& data_, std::vector<uint64_t>& keys) {
  for (size_t i = 0; i < data_.size(); ++i) {
    uint64_t uint64_value;
    std::memcpy(&uint64_value, &data_[i], sizeof(double));

    uint64_value = ((uint64_value >> 63) & 1) ? ~uint64_value : (uint64_value | (1ULL << 63));
    keys[i] = uint64_value;
  }
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskSequential::convert_uint64_to_doubles(
    const std::vector<uint64_t>& keys, std::vector<double>& data_) {
  for (size_t i = 0; i < keys.size(); ++i) {
    uint64_t uint64_value = keys[i];

    uint64_value = ((uint64_value >> 63) & 1) ? (uint64_value & ~(1ULL << 63)) : ~uint64_value;
    std::memcpy(&data_[i], &uint64_value, sizeof(double));
  }
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskSequential::radix_sort_uint64(
    std::vector<uint64_t>& keys) {
  constexpr int BITS = 64;
  constexpr int RADIX = 256;
  std::vector<uint64_t> temp(keys.size());

  for (int shift = 0; shift < BITS; shift += 8) {
    size_t count[RADIX + 1] = {0};

    for (uint64_t key : keys) {
      ++count[((key >> shift) & 255) + 1];
    }

    for (int i = 0; i < RADIX; ++i) {
      count[i + 1] += count[i];
    }

    for (uint64_t key : keys) {
      temp[count[(key >> shift) & 255]++] = key;
    }

    keys.swap(temp);
  }
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    int input_size = *(reinterpret_cast<int*>(taskData->inputs[0]));
    data.assign(reinterpret_cast<double*>(taskData->inputs[1]),
                reinterpret_cast<double*>(taskData->inputs[1]) + input_size);
  }

  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  bool is_valid = true;
  if (world.rank() == 0) {
    int input_size = *(reinterpret_cast<int*>(taskData->inputs[0]));

    if (taskData->inputs_count[0] != 1 || taskData->inputs_count[1] != static_cast<size_t>(input_size) ||
        taskData->outputs_count[0] != static_cast<size_t>(input_size)) {
      is_valid = false;
    }
  }

  mpi::broadcast(world, is_valid, 0);
  mpi::broadcast(world, data, 0);
  return is_valid;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  const int rank = world.rank();
  const int size = world.size();

  int local_size = data.size() / size;
  int remainder = data.size() % size;

  std::vector<int> data_sizes(size, local_size);
  std::vector<int> offsets(size);

  if (rank == 0) {
    for (int i = 0; i < remainder; ++i) {
      data_sizes[i]++;
    }

    offsets[0] = 0;
    for (int i = 1; i < size; ++i) {
      offsets[i] = offsets[i - 1] + data_sizes[i - 1];
    }
  }

  mpi::broadcast(world, data_sizes, 0);
  mpi::broadcast(world, offsets, 0);

  std::vector<double> local_data(data_sizes[rank]);
  mpi::scatterv(world, data.data(), data_sizes, offsets, local_data.data(), data_sizes[rank], 0);

  std::vector<uint64_t> local_keys(local_data.size());
  convert_doubles_to_uint64(local_data, local_keys);
  radix_sort_uint64(local_keys);
  convert_uint64_to_doubles(local_keys, local_data);

  int steps = std::log2(size) + 1;
  int group_size = 1;

  for (int step = 0; step < steps; ++step) {
    int partner_rank = rank + group_size;
    int group_half_size = group_size * 2;

    bool can_merge = (rank % group_half_size == 0);
    bool has_partner = (partner_rank < size);

    if (can_merge && has_partner) {
      int partner_data_size;
      world.recv(partner_rank, 0, partner_data_size);

      std::vector<double> partner_data(partner_data_size);
      world.recv(partner_rank, 1, partner_data.data(), partner_data_size);

      std::vector<double> merged_data;
      merged_data.reserve(local_data.size() + partner_data.size());
      std::merge(local_data.begin(), local_data.end(), partner_data.begin(), partner_data.end(),
                 std::back_inserter(merged_data));
      local_data.swap(merged_data);
    } else if (!can_merge && (rank % group_half_size == group_size)) {
      int receiver = rank - group_size;
      int current_data_size = local_data.size();

      world.send(receiver, 0, current_data_size);
      world.send(receiver, 1, local_data.data(), current_data_size);

      local_data.clear();
    }

    group_size *= 2;
  }

  if (rank == 0) {
    data.swap(local_data);
  }

  return true;
}


bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* out = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(data.begin(), data.end(), out);
  }

  return true;
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel::convert_doubles_to_uint64(
    const std::vector<double>& data_, std::vector<uint64_t>& keys) {
  for (size_t i = 0; i < data_.size(); ++i) {
    uint64_t uint64_value;
    std::memcpy(&uint64_value, &data_[i], sizeof(double));

uint64_value = ((uint64_value >> 63) & 1) != 0 ? ~uint64_value : (uint64_value | (1ULL << 63));
    keys[i] = uint64_value;
  }
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel::convert_uint64_to_doubles(
    const std::vector<uint64_t>& keys, std::vector<double>& data_) {
  for (size_t i = 0; i < keys.size(); ++i) {
    uint64_t uint64_value = keys[i];

uint64_value = ((uint64_value >> 63) & 1) != 0 ? (uint64_value & ~(1ULL << 63)) : ~uint64_value;
    std::memcpy(&data_[i], &uint64_value, sizeof(double));
  }
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel::radix_sort_uint64(
    std::vector<uint64_t>& keys) {
  constexpr int BITS = 64;
  constexpr int RADIX = 256;
  std::vector<uint64_t> temp(keys.size());

  for (int shift = 0; shift < BITS; shift += 8) {
    size_t count[RADIX + 1] = {0};

    for (uint64_t key : keys) {
      ++count[((key >> shift) & 255) + 1];
    }

    for (int i = 0; i < RADIX; ++i) {
      count[i + 1] += count[i];
    }

    for (uint64_t key : keys) {
      temp[count[(key >> shift) & 255]++] = key;
    }

    keys.swap(temp);
  }
}
