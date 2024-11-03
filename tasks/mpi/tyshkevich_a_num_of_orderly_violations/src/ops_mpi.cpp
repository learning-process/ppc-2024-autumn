#include "mpi/tyshkevich_a_num_of_orderly_violations/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> tyshkevich_a_num_of_orderly_violations_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool tyshkevich_a_num_of_orderly_violations_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  // Init vectors
  size = taskData->inputs_count[0];

  input_ = std::vector<int>(size);
  int *tmp_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
  for (int i = 0; i < size; i++) {
    input_[i] = tmp_ptr[i];
  }
  // Init values for output
  res = std::vector<int>(1, 0);

  return true;
}

bool tyshkevich_a_num_of_orderly_violations_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count of elements in I/O
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool tyshkevich_a_num_of_orderly_violations_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (int i = 1; i < size; i++) {
    if (input_[i - 1] > input_[i]) res[0]++;
  }
  return true;
}

bool tyshkevich_a_num_of_orderly_violations_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int *>(taskData->outputs[0])[0] = res[0];
  return true;
}

bool tyshkevich_a_num_of_orderly_violations_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  // Init vectors
  size = taskData->inputs_count[0];

  if (world.rank() == 0) {
    input_ = std::vector<int>(size);
    int *tmp_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
    for (int i = 0; i < size; i++) {
      input_[i] = tmp_ptr[i];
    }
    // Init values for output
    res = std::vector<int>(1, 0);
  }

  return true;
}

bool tyshkevich_a_num_of_orderly_violations_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  if (world.size() == 1) {
    for (int i = 1; i < size; i++) {
      if (input_[i - 1] > input_[i]) res[0]++;
    }
    return true;
  }

  int world_size = world.size();

  if (world.rank() > size) {
    return true;
  }
  if (world_size > size + 1) world_size = size + 1;

  int chunkSize = size / (world_size - 1);
  int lastChunkSize = size - chunkSize * (world_size - 2);

  if (world.rank() == 0) {
    for (int i = 0; i < world_size - 2; i++) {
      world.send(i + 1, 0, &input_.data() + i * chunkSize, chunkSize + 1);
    }
    world.send(world_size - 1, 0, &input_.data() + (world_size - 2) * lastChunkSize, lastChunkSize);

    int tempDef;
    for (int i = 0; i < world_size - 1; i++) {
      world.recv(i + 1, 1, &tempDef, 1);
      res[0] += tempDef;
    }
  } else {
    int localChunk = chunkSize + 1;
    if (world_size - 1 == world.rank()) localChunk = lastChunkSize;
    std::vector<int> chunk(localChunk);
    int counter = 0;
    world.recv(0, 0, chunk.data(), localChunk);

    for (int i = 1; i < localChunk; i++) {
      if (chunk[i - 1] > chunk[i]) counter++;
    }
    world.send(0, 1, &counter, 1);
  }

  return true;
}

bool tyshkevich_a_num_of_orderly_violations_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int *>(taskData->outputs[0])[0] = res[0];
  }
  return true;
}
