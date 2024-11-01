// Copyright 2023 Nesterov Alexander
#include "mpi/zinoviev_a_sum_cols_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dist(0, 100);
  std::vector<int> vec(sz);
  std::generate(vec.begin(), vec.end(), [&]() { return dist(gen); });
  return vec;
}

bool zinoviev_a_sum_cols_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  input_ = getRandomVector(100);  // например, 100 элементов
  return true;
}

bool zinoviev_a_sum_cols_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return true;
}

bool zinoviev_a_sum_cols_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  res = std::accumulate(input_.begin(), input_.end(), 0);
  return true;
}

bool zinoviev_a_sum_cols_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool zinoviev_a_sum_cols_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  std::cout << "Sequential Result: " << res << std::endl;
  return true;
}

bool zinoviev_a_sum_cols_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  input_ = getRandomVector(100);  // например, 100 элементов
  return true;
}

bool zinoviev_a_sum_cols_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int world_size = world.size();
  int world_rank = world.rank();

  // Разделяем вектор на части, каждая часть для своего процесса
  int chunk_size = input_.size() / world_size;
  local_input_ =
      std::vector<int>(input_.begin() + world_rank * chunk_size, input_.begin() + (world_rank + 1) * chunk_size);

  // Вычисляем сумму на локальном уровне
  int local_sum = std::accumulate(local_input_.begin(), local_input_.end(), 0);

  // Объединяем результаты с других процессов
  boost::mpi::reduce(world, local_sum, res, std::plus<int>(), 0);
  return true;
}

bool zinoviev_a_sum_cols_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    std::cout << "Parallel Result: " << res << std::endl;
  }
  return true;
}
