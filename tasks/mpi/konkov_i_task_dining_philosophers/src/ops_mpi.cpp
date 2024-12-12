#include "mpi/konkov_i_task_dining_philosophers/include/ops_mpi.hpp"

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

using namespace std::chrono_literals;

bool konkov_i_task_dining_philosophers::DiningPhilosophersMPITaskParallel::pre_processing() {
  internal_order_test();
  return true;
}

bool konkov_i_task_dining_philosophers::DiningPhilosophersMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool konkov_i_task_dining_philosophers::DiningPhilosophersMPITaskParallel::run() {
  internal_order_test();

  unsigned int delta = 0;

  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * delta, delta);
    }
  }
  broadcast(world, delta, 0);

  local_input_ = std::vector<int>(delta);
  if (world.rank() == 0) {
    std::copy(input_.begin(), input_.begin() + delta, local_input_.begin());
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }

  res = std::accumulate(local_input_.begin(), local_input_.end(), 0);

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
