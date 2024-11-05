
#include "seq/komshina_d_min_of_vector_elements/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);

  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }

  min_res = 0;
  return true;
}

bool komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential::run() {
  internal_order_test();
  int elementMin = input_[0];
  for (size_t i = 1; i < input_.size(); ++i) {
    if (input_[i] < elementMin) {
      elementMin = input_[i];
    }
  }
  min_res = elementMin;
  std::cout << "Min result: " << min_res << std::endl;  
  return true;
}

bool komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = min_res;
  return true;
}