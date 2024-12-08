// Copyright 2024 Kabalova Valeria
#include "mpi/kabalova_v_mpi_reduce/include/kabalova_mpi_reduce.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

// List of valid operations:
// + = MPI_SUM
// * = MPI_PROD
// min = MPI_MIN
// max = MPI_MAX
// && = MPI_LAND
// || = MPI_LOR
// & = MPI_BAND
// | = MPI_BOR
// ^ = MPI_BXOR
// lxor = MPI_LXOR

bool kabalova_v_mpi_reduce::checkValidOperation(const std::string& ops) {
  return (ops == "+" || ops == "*" || ops == "min" || ops == "max" || ops == "&&" || ops == "||" || ops == "&" ||
          ops == "|" || ops == "^" || ops == "lxor");
}

bool kabalova_v_mpi_reduce::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    // Init value for input
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
    local_input_ = std::vector<int>(taskData->inputs_count[0], 0);
  }
  return true;
}

bool kabalova_v_mpi_reduce::TestMPITaskParallel::validation() {
  internal_order_test();
  bool flag = true;
  if (world.rank() == 0) {
    flag = (taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1 && checkValidOperation(ops));
  }
  broadcast(world, flag, 0);
  return flag;
}

bool kabalova_v_mpi_reduce::TestMPITaskParallel::run() {
  internal_order_test();
  unsigned int delta = 0;
  unsigned int remains = 0;
  if (world.rank() == 0) {
    // Get delta = string.size() / num_threads
    delta = taskData->inputs_count[0] / world.size();
    remains = taskData->inputs_count[0] % world.size();
  }
  broadcast(world, delta, 0);
  broadcast(world, remains, 0);

  // Need proper arguments for scatterv
  std::vector<int> subvectorSizes(world.size(), delta);
  std::vector<int> offsets(world.size(), 0);
  if (world.rank() == 0) {
    for (unsigned int i = 0; i < remains; i++) subvectorSizes[i]++;
    for (int i = 1; i < world.size(); i++) offsets[i] = offsets[i - 1] + subvectorSizes[i - 1];

    unsigned int localSize = subvectorSizes[0];
    local_input_.resize(localSize);

    boost::mpi::scatterv(world, input_, subvectorSizes, offsets, local_input_.data(), localSize, 0);
  } else {
    if ((unsigned int)world.rank() < remains) {
      subvectorSizes[world.rank()]++;
    }
    unsigned int localSize = subvectorSizes[world.rank()];
    local_input_.resize(localSize);

    boost::mpi::scatterv(world, local_input_.data(), localSize, 0);
  }

  // After this we can finally use reduce
  int local_res;
  if (ops == "+") {  // MPI_SUM
    local_res = std::accumulate(local_input_.begin(), local_input_.end(), 0);
    reduce(world, local_res, result, std::plus(), 0);
  } else if (ops == "*") {  // MPI_PROD
    local_res = 1;
    for (size_t i = 0; i < local_input_.size(); i++) {
      local_res *= local_input_[i];
      if (local_res > std::numeric_limits<int>::max()) throw "overflow";
    }
    reduce(world, local_res, result, std::multiplies(), 0);
  } else if (ops == "max") {  // MPI_MAX
    local_res = *std::max_element(local_input_.begin(), local_input_.end());
    reduce(world, local_res, result, boost::mpi::maximum<int>(), 0);
  } else if (ops == "min") {  // MPI_MIN
    local_res = *std::min_element(local_input_.begin(), local_input_.end());
    reduce(world, local_res, result, boost::mpi::minimum<int>(), 0);
  } else if (ops == "&&") {  // MPI_LAND
    local_res = 1;
    for (size_t i = 0; i < local_input_.size(); i++) {
      local_res = static_cast<int>((bool)local_res && (bool)local_input_[i]);
    }
    reduce(world, local_res, result, std::logical_and(), 0);
  } else if (ops == "||") {  // MPI_LOR
    local_res = 0;
    for (size_t i = 0; i < local_input_.size(); i++) {
      local_res = static_cast<int>(bool(local_res) || bool(local_input_[i]));
    }
    reduce(world, local_res, result, std::logical_or(), 0);
  } else if (ops == "&") {  // MPI_BAND
    local_res = 1;
    for (size_t i = 0; i < local_input_.size(); i++) {
      local_res = local_res & local_input_[i];
    }
    reduce(world, local_res, result, boost::mpi::bitwise_and<int>(), 0);
  } else if (ops == "|") {  // MPI_BOR
    local_res = 0;
    for (size_t i = 0; i < local_input_.size(); i++) {
      local_res = local_res | local_input_[i];
    }
    reduce(world, local_res, result, boost::mpi::bitwise_or<int>(), 0);
  } else if (ops == "^") {  // MPI_BXOR
    local_res = 0;
    for (size_t i = 0; i < local_input_.size(); i++) {
      local_res = local_res ^ local_input_[i];
    }
    reduce(world, local_res, result, boost::mpi::bitwise_xor<int>(), 0);
  } else if (ops == "lxor") {  // MPI_LXOR
    local_res = 0;
    for (size_t i = 0; i < local_input_.size(); i++) {
      bool res1 = !(bool)local_res;
      bool res2 = !(bool)local_input_[i];
      local_res = static_cast<int>(res1 != res2);
    }
    reduce(world, local_res, result, boost::mpi::logical_xor<int>(), 0);
  } else
    throw("Wrong operation");
  return true;
}

bool kabalova_v_mpi_reduce::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
  }
  return true;
}