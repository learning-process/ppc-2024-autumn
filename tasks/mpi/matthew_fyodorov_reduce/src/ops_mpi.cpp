#include "mpi/matthew_fyodorov_reduce/include/ops_mpi.hpp"

#include <boost/mpi.hpp>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool matthew_fyodorov_reduce_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }

  res = 0;
  return true;
}

bool matthew_fyodorov_reduce_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  return taskData->outputs_count[0] == 1;
}

bool matthew_fyodorov_reduce_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  if (ops == "+") {
    res = std::accumulate(input_.begin(), input_.end(), 0);
  }
  return true;
}

bool matthew_fyodorov_reduce_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool matthew_fyodorov_reduce_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_.resize(taskData->inputs_count[0]);
    auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tempPtr, tempPtr + taskData->inputs_count[0], input_.begin());
  }
  res = 0;

  return true;
}

bool matthew_fyodorov_reduce_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool matthew_fyodorov_reduce_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int rank = world.rank();

  if (rank == 0) {
    for (int i = 0; i < (int)input_.size(); i += world.size()) {
      local_input_.push_back(input_[i]);
    }
    for (int proc_num = 1; proc_num < world.size(); proc_num++) {
      std::vector<int> local_data;
      for (int i = proc_num; i < (int)input_.size(); i += world.size()) {
        local_data.push_back(input_[i]);
      }
      world.send(proc_num, 0, local_data);
    }
  } else {
    world.recv(0, 0, local_input_);
  }
  int local_sum = std::accumulate(local_input_.begin(), local_input_.end(), 0);
  int global_sum = 0;

  boost::mpi::reduce(world, local_sum, global_sum, std::plus<>(), 0);
  if (rank == 0) res = global_sum;

  return true;
}

bool matthew_fyodorov_reduce_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
