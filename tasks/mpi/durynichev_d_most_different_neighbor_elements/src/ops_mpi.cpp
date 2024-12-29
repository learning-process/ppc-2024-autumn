#include "mpi/durynichev_d_most_different_neighbor_elements/include/ops_mpi.hpp"

bool durynichev_d_most_different_neighbor_elements_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] >= 2 && taskData->outputs_count[0] == 2;
}

bool durynichev_d_most_different_neighbor_elements_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  auto *input_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
  auto input_size = taskData->inputs_count[0];
  input.assign(input_ptr, input_ptr + input_size);
  result.resize(2);
  return true;
}

bool durynichev_d_most_different_neighbor_elements_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  result[0] = input[0];
  result[1] = input[1];
  int maxDiff = 0;

  for (size_t i = 1; i < input.size(); ++i) {
    int diff = std::abs(input[i] - input[i - 1]);
    if (diff > maxDiff) {
      maxDiff = diff;
      result[0] = std::min(input[i], input[i - 1]);
      result[1] = std::max(input[i], input[i - 1]);
    }
  }
  return true;
}

bool durynichev_d_most_different_neighbor_elements_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  std::copy_n(result.begin(), 2, reinterpret_cast<int *>(taskData->outputs[0]));
  return true;
}

bool durynichev_d_most_different_neighbor_elements_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] >= 2 && taskData->outputs_count[0] == 2;
  }
  return true;
}

bool durynichev_d_most_different_neighbor_elements_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  int *input_ptr{};
  int input_size{};
  if (world.rank() == 0) {
    input_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
    input_size = taskData->inputs_count[0];
    input.assign(input_ptr, input_ptr + input_size);
  }
  boost::mpi::broadcast(world, input_size, 0);

  int chunk_size = input_size / world.size();
  int extra_chunks = input_size % world.size();
  int used = std::min(world.size(), input_size);

  std::vector<int> all_chunk_sizes(world.size(), 0);
  std::fill(all_chunk_sizes.begin(), all_chunk_sizes.begin() + used, chunk_size);
  for (int i = 0; i < extra_chunks; i++) {
    all_chunk_sizes.at(i)++;
  }
  std::vector<int> chunk_offsets(world.size(), 0);
  for (int i = 1; i < used; i++) {
    chunk_offsets[i] = chunk_offsets[i - 1] + all_chunk_sizes[i];
  }
  for (int i = 0; i < used - 1; i++) {
    all_chunk_sizes.at(i)++;
  }

  chunkStart = chunk_offsets[world.rank()];
  chunk.resize(all_chunk_sizes[world.rank()]);
  boost::mpi::scatterv(world, input.data(), all_chunk_sizes, chunk_offsets, chunk.data(), chunk.size(), 0);

  return true;
}

bool durynichev_d_most_different_neighbor_elements_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  ChunkResult chunk_result{0, 0, std::numeric_limits<int>::min()};
  if (chunk.size() >= 2) {
    chunk_result = ChunkResult{chunkStart, chunkStart + 1, std::abs(chunk[0] - chunk[1])};
    for (size_t i = 1; i < chunk.size(); ++i) {
      int diff = std::abs(chunk[i] - chunk[i - 1]);
      if (diff > chunk_result.diff) {
        chunk_result = ChunkResult{i - 1 + chunkStart, i + chunkStart, diff};
      }
    }
  }
  boost::mpi::reduce(world, chunk_result, result, ChunkResult(), 0);
  return true;
}

bool durynichev_d_most_different_neighbor_elements_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto vec = result.toVector(input);
    std::copy_n(vec.begin(), 2, reinterpret_cast<int *>(taskData->outputs[0]));
  }
  return true;
}
