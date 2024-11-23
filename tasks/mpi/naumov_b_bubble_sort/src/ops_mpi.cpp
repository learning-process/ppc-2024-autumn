// Copyright 2023 Nesterov Alexander
#include "mpi/naumov_b_bubble_sort/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

std::vector<int> naumov_b_bubble_sort_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool naumov_b_bubble_sort_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  return true;
}

bool naumov_b_bubble_sort_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  return true;
}

bool naumov_b_bubble_sort_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  return true;
}

bool naumov_b_bubble_sort_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test(); 
  return true;
}
