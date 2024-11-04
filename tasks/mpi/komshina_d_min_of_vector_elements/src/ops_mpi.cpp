#include "mpi/komshina_d_min_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> generateRandomVector(int size) {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = generator() % 100;
  }
  return vec;
}

bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementsTaskSequential::pre_processing() {
  internal_order_test();
 
  input_ = std::vector<int>(taskData->inputs_count[0]);
  int* inputPtr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(inputPtr, inputPtr + taskData->inputs_count[0], input_.begin());
  min_res = 0; 
  return true;
}

bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementsTaskSequential::validation() {
  internal_order_test();
 
  return taskData->outputs_count[0] == 1;
}

bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementsTaskSequential::run() {
  internal_order_test();

  min_res = *std::min_element(input_.begin(), input_.end());
  return true;
}

bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementsTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = min_res;
  return true;
}

bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementsTaskParallel::pre_processing() {
  internal_order_test();
  unsigned int partition_size = 0;
  if (world.rank() == 0) {
    partition_size = taskData->inputs_count[0] / world.size();
  }
  broadcast(world, partition_size, 0);

  if (world.rank() == 0) {
   
    input_ = std::vector<int>(taskData->inputs_count[0]);
    int* inputPtr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(inputPtr, inputPtr + taskData->inputs_count[0], input_.begin());
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * partition_size, partition_size);
    }
  }

  local_input_ = std::vector<int>(partition_size);
  if (world.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + partition_size);
  } else {
    world.recv(0, 0, local_input_.data(), partition_size);
  }

  min_res = std::numeric_limits<int>::max(); 
  return true;
}

bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementsTaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementsTaskParallel::run() {
  internal_order_test();
  int local_min = *std::min_element(local_input_.begin(), local_input_.end());
  reduce(world, local_min, min_res, boost::mpi::minimum<int>(), 0);
  return true;
}

bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementsTaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = min_res;
  }
  return true;
}
