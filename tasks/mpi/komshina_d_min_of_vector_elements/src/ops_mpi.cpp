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
    vec[i] = gen() % 100;
  }
  return vec;
}

bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<int>(taskData->inputs_count[0]);
  int* ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (size_t i = 0; i < taskData->inputs_count[0]; ++i) {
    input_[i] = ptr[i];
  }
  res = input_[0];
  return true;
}

bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs_count[0] == 0) {
    return false;
  }

  if (taskData->outputs_count[0] != 1) {
    return false;
  }
  return true;
}

bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskSequential::run() {
  internal_order_test();
  res = input_[0];
  for (size_t ptr = 1; ptr < input_.size(); ++ptr) {
    if (res > input_[ptr]) {
      res = input_[ptr];
    }
  }
  return true;
}

bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_ = std::vector<int>(taskData->inputs_count[0]);
    int* ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (size_t i = 0; i < taskData->inputs_count[0]; ++i) {
      input_[i] = ptr[i];
    }
  }
  return true;
}

bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskParallel::validation() {
  internal_order_test();
  if (taskData->inputs_count[0] == 0) {
    return false;
  }

  if (taskData->outputs_count[0] != 1) {
    return false;
  }
  return true;
}

bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskParallel::run() {
  internal_order_test();

  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
    int remainder = taskData->inputs_count[0] % world.size();
    if (remainder > 0) {
      delta++;
    }
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * delta, delta);
    }
  }

  if (world.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta);
  } else {
    local_input_ = std::vector<int>(delta, INT_MAX);
    world.recv(0, 0, local_input_.data(), delta);
  }

  int local_res = INT_MAX;
  if (!local_input_.empty()) {
    local_res = local_input_[0];
    for (const auto& i : local_input_) {
      if (local_res > i) {
        local_res = i;
      }
    }
  }

  reduce(world, local_res, res, boost::mpi::minimum<int>(), 0);

  return true;
}

bool komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
