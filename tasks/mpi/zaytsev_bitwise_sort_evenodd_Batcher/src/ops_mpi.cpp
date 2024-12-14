// Copyright 2023 Nesterov Alexander
#include "mpi/zaytsev_bitwise_sort_evenodd_Batcher/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <vector>

bool zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  return true;
}

bool zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskSequential::validation() {
  internal_order_test();
  return true;
}

bool zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskSequential::run() {
  internal_order_test();
  return true;
}

bool zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskSequential::post_processing() {
  internal_order_test();
  return true;
}

bool zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  return true;
}

bool zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskParallel::validation() {
  internal_order_test();
  return true;
}

bool zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskParallel::run() {
  internal_order_test();
  return true;
}

bool zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskParallel::post_processing() {
  internal_order_test();
  return true;
}