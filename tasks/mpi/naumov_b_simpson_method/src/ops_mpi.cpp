// Copyright 2023 Nesterov Alexander
#include "mpi/naumov_b_simpson_method/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>


bool naumov_b_simpson_method_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  
  return true;
}

bool naumov_b_simpson_method_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
 
  return true;
}

bool naumov_b_simpson_method_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  
  return true;
}

bool naumov_b_simpson_method_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  
  return true;
}

bool naumov_b_simpson_method_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  
  return true;
}

bool naumov_b_simpson_method_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  
  return true;
}

bool naumov_b_simpson_method_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  
  return true;
}

bool naumov_b_simpson_method_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  
  return true;
}
