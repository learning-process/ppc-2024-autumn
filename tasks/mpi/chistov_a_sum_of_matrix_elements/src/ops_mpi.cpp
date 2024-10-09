#include "mpi/chistov_a_sum_of_matrix_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <stdexcept>

using namespace std::chrono_literals;


int chistov_a_sum_of_matrix_elements::classic_way(const std::vector<int> matrix, int n, int m) {
  int result = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      result += matrix[i * m + j];  
    }
  }

  return result;
}

void chistov_a_sum_of_matrix_elements::print_matrix(std::vector<int> matrix, int n, int m) { 
  std::cout << "Matrix:" << std::endl;
  for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) std::cout << matrix[i * m + j] << " ";
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

std::vector<int> chistov_a_sum_of_matrix_elements::getRandomMatrix(int n, int m) {
  if (n <= 0 || m <= 0) {
    throw std::invalid_argument("Incorrect entered N or M");
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distribution(-100, 100);
  std::vector<int> matrix(n * m);

  for (int i = 0; i < n * m; ++i) {
    matrix[i] = static_cast<int>(distribution(gen));
  }

  return matrix;
}


bool chistov_a_sum_of_matrix_elements::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  int* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  input_.resize(taskData->inputs_count[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());

  return true;
}

bool chistov_a_sum_of_matrix_elements::TestMPITaskSequential::validation() {
  internal_order_test();

  return taskData->outputs_count[0] == 1; //we  return one element
}

bool chistov_a_sum_of_matrix_elements::TestMPITaskSequential::run() {
  internal_order_test();

  res = std::accumulate(input_.begin(), input_.end(), 0);
  return true;
}

bool chistov_a_sum_of_matrix_elements::TestMPITaskSequential::post_processing() {
  internal_order_test();

  if (taskData->outputs.size() > 0 && taskData->outputs[0] != nullptr) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
    return true;
  } 
  else {
    return false;
  }
}

// Parallel

bool chistov_a_sum_of_matrix_elements::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    // Init vectors
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * delta, delta);
    }
  }
  local_input_ = std::vector<int>(delta);
  if (world.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta);
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }
  // Init value for output
  res = 0;
  return true;
}


bool chistov_a_sum_of_matrix_elements::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}



bool chistov_a_sum_of_matrix_elements::TestMPITaskParallel::run() {
  internal_order_test();

  int local_res;
  local_res = std::accumulate(local_input_.begin(), local_input_.end(), 0);
  reduce(world, local_res, res, std::plus(), 0);
  
  return true;
}

bool chistov_a_sum_of_matrix_elements::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
