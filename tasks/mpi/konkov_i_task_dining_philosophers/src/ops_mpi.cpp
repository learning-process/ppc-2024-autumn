#include "mpi/konkov_i_task_dining_philosophers/include/ops_mpi.hpp"

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> konkov_i_task_dining_philosophers::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool konkov_i_task_dining_philosophers::DiningPhilosophersMPITaskParallel::pre_processing() {
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

bool konkov_i_task_dining_philosophers::DiningPhilosophersMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool konkov_i_task_dining_philosophers::DiningPhilosophersMPITaskParallel::run() {
  internal_order_test();
  // Simulate dining philosophers in parallel
  std::vector<std::mutex> forks(local_input_.size());
  for (int i = 0; i < local_input_.size(); ++i) {
    std::unique_lock<std::mutex> left_fork(forks[i]);
    std::unique_lock<std::mutex> right_fork(forks[(i + 1) % local_input_.size()]);
    // Philosopher is eating
    res += local_input_[i];
  }

  int local_res = res;
  reduce(world, local_res, res, std::plus(), 0);

  return true;
}

bool konkov_i_task_dining_philosophers::DiningPhilosophersMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}