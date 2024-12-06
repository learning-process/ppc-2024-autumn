// Copyright 2023 Nesterov Alexander
#include "mpi/prokhorov_n_producer_customer/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace prokhorov_n_producer_customer_mpi {
std::vector<int> getRandomVector(int sz) {
  std::vector<int> vec(sz);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(1, 100);
  for (int &val : vec) val = dist(gen);
  return vec;
}

bool TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.size() < 2) {
    return false;
  }

  unsigned int total_data_count = 0;
  unsigned int range_start = 0, range_end = 0;

  if (world.rank() == 0) {
    if (producer_data.empty()) {
      producer_data = getRandomVector(100);
    }

    total_data_count = producer_data.size();
    if (total_data_count == 0) {
      return false;
    }
  }

  broadcast(world, total_data_count, 0);

  unsigned int delta = total_data_count / (world.size() - 1);
  range_start = (world.rank() - 1) * delta;
  range_end = (world.rank() == world.size() - 1) ? total_data_count : range_start + delta;

  if (world.rank() != 0) {
    local_input_.resize(range_end - range_start);
  }

  if (world.rank() == 0) {
    for (int rank = 1; rank < world.size(); ++rank) {
      unsigned int start = (rank - 1) * delta;
      unsigned int end = (rank == world.size() - 1) ? total_data_count : start + delta;
      world.send(rank, 0, producer_data.data() + start, end - start);
    }
  } else {
    world.recv(0, 0, local_input_.data(), range_end - range_start);
  }

  return true;
}

bool TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (producer_data.empty() || taskData->outputs_count[0] != 1) {
      return false;
    }
  }

  if (local_input_.empty()) {
    return false;
  }

  return true;
}

bool TestMPITaskParallel::run() {
  internal_order_test();

  if (world.rank() == 0) {
    if (producer_data.empty()) {
      return false;
    }
    int data_size = producer_data.size();
    if (data_size <= 0) {
      return false;
    }
    broadcast(world, data_size, 0);

    int delta = data_size / (world.size() - 1);
    if (delta <= 0) {
      return false;
    }

    for (int i = 0; i < world.size() - 1; ++i) {
      int start = i * delta;
      int end = (i == world.size() - 2) ? data_size : (i + 1) * delta;
      world.send(i + 1, 0, producer_data.data() + start, end - start);
    }
  }

  if (world.rank() != 0) {
    int data_size;
    broadcast(world, data_size, 0);
    if (data_size <= 0) {
      return false;
    }

    local_input_.resize(data_size);
    world.recv(0, 0, local_input_.data(), data_size);
  }

  return true;
}

bool prokhorov_n_producer_customer_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->outputs_count[0] != 1) {
      return false;
    }

    int total_result = res;
    for (int i = 1; i < world.size(); ++i) {
      int local_result = 0;
      world.recv(i, 0, &local_result, 1);
      total_result += local_result;
    }

    reinterpret_cast<int *>(taskData->outputs[0])[0] = total_result;
  }

  return true;
}
