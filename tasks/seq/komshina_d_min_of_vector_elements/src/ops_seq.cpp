#include "seq/komshina_d_min_of_vector_elements/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool komshina_d_min_of_vector_elements_seq::MinOfVectorElementsSeq::pre_processing() {
  internal_order_test();
  input_vector_ = std::vector<int>(taskData->inputs_count[0]);
  int* it = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(it, it + taskData->inputs_count[0], input_vector_.begin());
  res = input_vector_[0];  
  return true;
}

bool komshina_d_min_of_vector_elements_seq::MinOfVectorElementsSeq::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] > 0) && (taskData->outputs_count[0] == 1);
}

bool komshina_d_min_of_vector_elements_seq::MinOfVectorElementsSeq::run() {
  internal_order_test();
  for (size_t it = 1; it < input_vector_.size(); ++it) {
    if (res > input_vector_[it]) {
      res = input_vector_[it];  
    }
  }
  return true;
}

bool komshina_d_min_of_vector_elements_seq::MinOfVectorElementsSeq::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;  
  return true;
}
