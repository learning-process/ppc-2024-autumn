
#include "mpi/komshina_d_min_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> komshina_d_min_of_vector_elements_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 1000;
  }
  return vec;
}

bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskSequential::pre_processing() {
  internal_order_test();
 
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);

  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) 
  {
    input_[i] = tmp_ptr[i];
  }

  min_res = 0;
  return true;
}

bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskSequential::validation() {
  internal_order_test();
  
  return (taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1);
}

bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskSequential::run() {
  internal_order_test();

  int elementMin = input_[0];  
  for (size_t i = 1; i < input_.size(); ++i) {  
    if (input_[i] < elementMin) {
      elementMin = input_[i];  
    }
  }
  min_res = elementMin; 
  return true;  
}

bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = min_res;
  return true;
}





bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = taskData->inputs_count[0] / world.size();  
  int remainder = taskData->inputs_count[0] % world.size(); 

 
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);

    
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());

    
    for (int proc = 1; proc < world.size(); proc++) {
      int current_delta = delta + (proc <= remainder ? 1 : 0); 
      world.send(proc, 0, input_.data() + proc * delta, current_delta);  
    }
  }

   local_input_.resize(delta + (world.rank() <= remainder ? 1 : 0));
  if (world.rank() == 0) {
    std::copy(input_.begin(), input_.begin() + delta, local_input_.begin());
  } else {
    world.recv(0, 0, local_input_.data(), delta + (world.rank() <= remainder ? 1 : 0));
  }

  min_res = 0;
  return true;
}


bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    
    return taskData->outputs_count[0] == 1;
  }
  return true;
}
bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskParallel::run() {
  internal_order_test();

  int local_min = INT_MAX;
  for (int i : local_input_) {
    if (i < local_min) {
      local_min = i;
    }
  }

  reduce(world, local_min, min_res, boost::mpi::minimum<int>(), 0);

  return true;
}


bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = min_res;
  }
  return true;
}