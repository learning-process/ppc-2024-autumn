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

  if (world.rank() == 0) {
    auto input_size = taskData->inputs_count[0];
    auto chunk_size = input_size / world.size();
    std::cout << chunk_size << std::endl;
    auto *input_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
    input.assign(input_ptr, input_ptr + input_size);
    chunk.assign(input_ptr, input_ptr + chunk_size + int(world.size() > 1));

    for (int proc = 1; proc < world.size(); proc++) {
      auto start = proc * chunk_size;
      auto size = (proc == world.size() - 1) ? input_size - start : chunk_size + 1;
      world.send(proc, 0, std::vector<int>(input_ptr + start, input_ptr + start + size));
      world.send(proc, 1, start);
    }

  } else {
    world.recv(0, 0, chunk);
    world.recv(0, 1, chunkStart);
  }

  return true;
}

bool durynichev_d_most_different_neighbor_elements_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  auto chunk_result = ChunkResult{0, 1, std::abs(chunk[0] - chunk[1])};
  for (size_t i = 2; i < chunk.size(); ++i) {
    int diff = std::abs(chunk[i] - chunk[i - 1]);
    if (diff > chunk_result.diff) {
      chunk_result = ChunkResult{i - 1 + chunkStart, i + chunkStart, diff};
    }
  }
  boost::mpi::reduce(world, chunk_result, result, ChunkResult(), 0);
  return true;
}

bool durynichev_d_most_different_neighbor_elements_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    std::copy_n(result.toVector(input).begin(), 2, reinterpret_cast<int *>(taskData->outputs[0]));
  }
  return true;
}
