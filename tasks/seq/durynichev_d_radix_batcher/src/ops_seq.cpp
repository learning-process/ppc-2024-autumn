#include "seq/durynichev_d_radix_batcher/include/ops_seq.hpp"

#include <algorithm>

namespace durynichev_d_radix_batcher_seq {

bool RadixBatcher::validation() {
  internal_order_test();

  return taskData->inputs_count[0] > 1;
}

bool RadixBatcher::pre_processing() {
  internal_order_test();

  auto* vec_data = reinterpret_cast<double*>(taskData->inputs[0]);
  int vec_size = taskData->inputs_count[0];

  arr.assign(vec_data, vec_data + vec_size);

  return true;
}

bool RadixBatcher::run() {
  internal_order_test();

  radixSortDouble(arr.begin(), arr.end());

  return true;
}

bool RadixBatcher::post_processing() {
  internal_order_test();

  auto* out_vector_ = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(arr.begin(), arr.end(), out_vector_);

  return true;
}

}  // namespace durynichev_d_radix_batcher_seq
