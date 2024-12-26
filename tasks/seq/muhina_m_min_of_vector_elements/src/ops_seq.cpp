// Copyright 2024 Nesterov Alexander
#include "seq/muhina_m_min_of_vector_elements/include/ops_seq.hpp"

#include <random>
#include <thread>

using namespace std::chrono_literals;

int muhina_m_min_of_vector_elements_seq::vectorMin(std::vector<int, std::allocator<int>> vect) {
  int mini = vect[0];

  for (size_t i = 1; i < vect.size(); i++) {
    if (vect[i] < mini) {
      mini = vect[i];
    }
  }
  return mini;
}

bool muhina_m_min_of_vector_elements_seq::MinOfVectorSequential::pre_processing() {
  internal_order_test();

  // Init data vector
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tempPtr[i];
  }

  return true;
}

bool muhina_m_min_of_vector_elements_seq::MinOfVectorSequential::validation() {
  internal_order_test();
  // Handle empty input vector
  if (taskData->inputs_count[0] == 0) {
    return taskData->outputs_count[0] == 0;
  }
  return taskData->outputs_count[0] == 1;
}

bool muhina_m_min_of_vector_elements_seq::MinOfVectorSequential::run() {
  internal_order_test();
  // Iterate through the vector
  res_ = muhina_m_min_of_vector_elements_seq::vectorMin(input_);
  return true;
}

bool muhina_m_min_of_vector_elements_seq::MinOfVectorSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  return true;
}
