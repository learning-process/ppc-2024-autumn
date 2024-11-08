#include "seq/vladimirova_j_max_of_vector_elements/include/ops_seq.hpp"

#include <random>
#include <thread>

using namespace std::chrono_literals;

int vladimirova_j_max_of_vector_elements_seq::FindMaxElem(std::vector<int> m) {
  if (m.empty()) return INT_MIN;
  int max_elem = m[0];
  for (int& i : m) {
    if (i > max_elem) {
      max_elem = i;
    }
  }
  return max_elem;
}

bool vladimirova_j_max_of_vector_elements_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  input_ = std::vector<int>(taskData->inputs_count[0] * taskData->inputs_count[1]);

  for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
    auto* input_data = reinterpret_cast<int*>(taskData->inputs[i]);
    for (unsigned int j = 0; j < taskData->inputs_count[1]; j++) {
      input_[i * taskData->inputs_count[1] + j] = input_data[j];
    }
  }
  return true;
}

bool vladimirova_j_max_of_vector_elements_seq::TestTaskSequential::validation() {
  internal_order_test();

  return ((taskData->inputs_count[0] > 0) && (taskData->inputs_count[1] > 0)) && (taskData->outputs_count[0] == 1);
}

bool vladimirova_j_max_of_vector_elements_seq::TestTaskSequential::run() {
  internal_order_test();

  res = vladimirova_j_max_of_vector_elements_seq::FindMaxElem(input_);
  return true;
}

bool vladimirova_j_max_of_vector_elements_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
