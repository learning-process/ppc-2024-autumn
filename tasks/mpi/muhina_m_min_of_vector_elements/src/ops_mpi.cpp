// Copyright 2023 Nesterov Alexander
#include "mpi/muhina_m_min_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> muhina_m_min_of_vector_elements_mpi::GetRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 200 - 100;  // Generate random numbers from -100 to 99
  }
  return vec;
}

int muhina_m_min_of_vector_elements_mpi::vectorMin(std::vector<int, std::allocator<int>> vect) {
  if (vect.empty()) {
    return std::numeric_limits<int>::max();  // Handle empty vectors
  }
  int mini = vect[0];

  for (std::string::size_type i = 1; i < vect.size(); i++) {  
    if (vect[i] < mini) {
      mini = vect[i];
    }
  }
  return mini;
}

bool muhina_m_min_of_vector_elements_mpi::MinOfVectorMPISequential::pre_processing() {
  internal_order_test();
  // Init vectors
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  return true;
}

bool muhina_m_min_of_vector_elements_mpi::MinOfVectorMPISequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1;
}

bool muhina_m_min_of_vector_elements_mpi::MinOfVectorMPISequential::run() {
  internal_order_test();
  if (input_.empty()) {
    // Handle the case when the input vector is empty
    return true;
  }
  res_ = muhina_m_min_of_vector_elements_mpi::vectorMin(input_);
  return true;
}

bool muhina_m_min_of_vector_elements_mpi::MinOfVectorMPISequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  return true;
}

bool muhina_m_min_of_vector_elements_mpi::MinOfVectorMPIParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;
  if (world_.rank() == 0) {
    delta = taskData->inputs_count[0] / world_.size();
  }
  broadcast(world_, delta, 0);

  if (world_.rank() == 0) {
    // Init vectors
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
    for (int proc = 1; proc < world_.size(); proc++) {
      world_.send(proc, 0, input_.data() + proc * delta, delta);
    }
  }
  local_input_ = std::vector<int>(delta);
  if (world_.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta);
  } else {
    world_.recv(0, 0, local_input_.data(), delta);
  }
  return true;
}

bool muhina_m_min_of_vector_elements_mpi::MinOfVectorMPIParallel::validation() {
  internal_order_test();
  if (world_.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool muhina_m_min_of_vector_elements_mpi::MinOfVectorMPIParallel::run() {
  internal_order_test();
  if (local_input_.empty()) {
    // Handle the case when the local input vector is empty
    return true;
  }
  int local_min = muhina_m_min_of_vector_elements_mpi::vectorMin(local_input_);

  reduce(world_, local_min, res_, boost::mpi::minimum<int>(), 0);
  return true;
}

bool muhina_m_min_of_vector_elements_mpi::MinOfVectorMPIParallel::post_processing() {
  internal_order_test();
  if (world_.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  }
  return true;
}
