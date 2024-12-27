#include "seq/lavrentyev_a_radix_sort_simple_merge/include/ops_seq.hpp"

#include <algorithm>

namespace lavrentyev_a_radix_sort_simple_merge_seq {

bool RadixSimpleMerge::validation() {
  internal_order_test();

  return taskData->inputs_count[0] > 1;
}

bool RadixSimpleMerge::pre_processing() {
  internal_order_test();

  auto* vec_data = reinterpret_cast<double*>(taskData->inputs[0]);
  vsize = taskData->inputs_count[0];

  arr.assign(vec_data, vec_data + vsize);

  return true;
}

bool RadixSimpleMerge::run() {
  internal_order_test();

  radixSortDouble(arr.begin(), arr.end());

  return true;
}

bool RadixSimpleMerge::post_processing() {
  internal_order_test();

  auto* out_vector_ = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(arr.begin(), arr.end(), out_vector_);

  return true;
}

}  // namespace lavrentyev_a_radix_sort_simple_merge_seq
