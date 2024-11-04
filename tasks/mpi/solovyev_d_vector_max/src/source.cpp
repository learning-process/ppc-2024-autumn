#include <algorithm>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "mpi/solovyev_d_vector_max/include/header.hpp"

int solovyev_d_vector_max_mpi::vectorMax(std::vector<int, std::allocator<int>> v) {
  int m = std::numeric_limits<int>::min();
  for (std::string::size_type i = 0; i < v.size(); i++) {
    if (v[i] > m) {
      m = v[i];
    }
  }
  return m;
}

bool solovyev_d_vector_max_mpi::VectorMaxMPIParallel::pre_processing() {
  internal_order_test();

  // Determine number of vector elements per process
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
  }

  // Share delta between all processes
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    // Convert input data to vector
    int* input_ = reinterpret_cast<int*>(taskData->inputs[0]);
    data = std::vector<int>(input_, input_ + taskData->inputs_count[0]);

    // Send each of processes their portion of data
    for (int process = 1; process < world.size(); process++) {
      world.send(process, 0, data.data() + process * delta, delta);
    }
  }

  // Initialize local vector
  localData = std::vector<int>(delta);
  if (world.rank() == 0) {
    // Getting data directly if we in zero process
    localData = std::vector<int>(data.begin(), data.begin() + delta);
  } else {
    // Otherwise, recieving data
    world.recv(0, 0, localData.data(), delta);
  }

  // Init result value
  result = 0;
  return true;
}

bool solovyev_d_vector_max_mpi::VectorMaxMPIParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return (taskData->outputs_count[0] == 1 and taskData->inputs_count[0] != 0);
  }
  return true;
}

bool solovyev_d_vector_max_mpi::VectorMaxMPIParallel::run() {
  internal_order_test();
  int localResult;

  // Search for maximum vector element in current process data
  localResult = vectorMax(localData);

  // Search for maximum vector element using all processes data
  reduce(world, localResult, result, boost::mpi::maximum<int>(), 0);
  return true;
}

bool solovyev_d_vector_max_mpi::VectorMaxMPIParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
  }
  return true;
}

bool solovyev_d_vector_max_mpi::VectorMaxSequential::pre_processing() {
  internal_order_test();

  // Init data vector
  int* input_ = reinterpret_cast<int*>(taskData->inputs[0]);
  data = std::vector<int>(input_, input_ + taskData->inputs_count[0]);

  // Init result value
  result = 0;
  return true;
}

bool solovyev_d_vector_max_mpi::VectorMaxSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return (taskData->outputs_count[0] == 1 and taskData->inputs_count[0] != 0);
}

bool solovyev_d_vector_max_mpi::VectorMaxSequential::run() {
  internal_order_test();

  // Determine maximum value of data vector
  result = vectorMax(data);
  return true;
}

bool solovyev_d_vector_max_mpi::VectorMaxSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
  return true;
}