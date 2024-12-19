// Copyright 2024 Nesterov Alexander
#include "seq/naumov_b_simpson_method/include/ops_seq.hpp"

#include <thread>



bool naumov_b_simpson_method_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  
  return true;
}

bool naumov_b_simpson_method_seq::TestTaskSequential::validation() {
  internal_order_test();

  return true;
}

bool naumov_b_simpson_method_seq::TestTaskSequential::run() {
  internal_order_test();
  
  return true;
}

bool naumov_b_simpson_method_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  
  return true;
}
