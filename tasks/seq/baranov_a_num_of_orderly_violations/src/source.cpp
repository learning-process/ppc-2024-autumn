#include "seq/baranov_a_num_of_orderly_violations/include/header.hpp"
namespace baranov_a_num_of_orderly_violations_seq {

template <typename iotype, typename cntype>
cntype num_of_orderly_violations<iotype, cntype>::seq_proc(std::vector<iotype> vec) {
  cntype num = 0;
  int n = vec.size();
  for (int i = 0; i < n - 1; ++i) {
    if (vec[i] < vec[i + 1]) {
      ++num;
    }
  }
  return num;
}

template <class iotype, class cntype>
bool num_of_orderly_violations<iotype, cntype>::pre_processing() {
  internal_order_test();
  // Init vectors
  int n = taskData->inputs_count[0];
  input_ = std::vector<iotype>(n);
  void* ptr_r = taskData->inputs[0];
  void* ptr_d = input_.data();
  memcpy(ptr_d, ptr_r, sizeof(iotype) * n);
  // Init value for output
  num_ = 0;
  return true;
}
template <class iotype, class cntype>
bool num_of_orderly_violations<iotype, cntype>::validation() {
  internal_order_test();
  // Check count elements of output

  if (taskData->outputs_count[0] == 1) {
    return true;
  }
  return false;
}
template <class iotype, class cntype>
bool num_of_orderly_violations<iotype, cntype>::run() {
  internal_order_test();
  num_ = seq_proc(input_);

  return true;
}
template <class iotype, class cntype>
bool baranov_a_num_of_orderly_violations_seq::num_of_orderly_violations<iotype, cntype>::post_processing() {
  internal_order_test();
  reinterpret_cast<cntype*>(taskData->outputs[0])[0] = num_;
  return true;
}

template class baranov_a_num_of_orderly_violations_seq::num_of_orderly_violations<int, int>;

template class baranov_a_num_of_orderly_violations_seq::num_of_orderly_violations<double, int>;
}  // namespace baranov_a_num_of_orderly_violations_seq