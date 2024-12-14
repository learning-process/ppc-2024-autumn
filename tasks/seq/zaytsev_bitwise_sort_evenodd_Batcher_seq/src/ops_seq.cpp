// Copyright 2024 Nesterov Alexander
#include "seq/zaytsev_bitwise_sort_evenodd_Batcher_seq/include/ops_seq.hpp"

bool zaytsev_bitwise_sort_evenodd_Batcher_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  
  return true;
}

bool zaytsev_bitwise_sort_evenodd_Batcher_seq::TestTaskSequential::validation() {
  internal_order_test();

  return true;
}

bool zaytsev_bitwise_sort_evenodd_Batcher_seq::TestTaskSequential::run() {
  internal_order_test();

  return true;
}

bool zaytsev_bitwise_sort_evenodd_Batcher_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  
  return true;
}
