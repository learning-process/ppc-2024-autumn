#include "mpi/mironov_a_broadcast/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace boost::mpi;

bool mironov_a_broadcast_mpi::ComponentSumPowerBoostImpl::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int size1 = taskData->inputs_count[0];
    int size2 = taskData->inputs_count[1];
    delta_ = size2 / world.size();
    if (size2 % world.size() != 0) {
      delta_++;
    }
    size_ = size1;
    input_ = std::vector<int>(size1);
    powers_ = std::vector<int>(delta_ * world.size(), -1);

    int* array_it = reinterpret_cast<int*>(taskData->inputs[0]);
    int* powers_it = reinterpret_cast<int*>(taskData->inputs[1]);
    std::copy(array_it, array_it + size1, input_.begin());
    std::copy(powers_it, powers_it + size2, powers_.begin());
  }
  return true;
}

bool mironov_a_broadcast_mpi::ComponentSumPowerBoostImpl::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (taskData->inputs_count[0] > 0) && (taskData->inputs_count[1] > 0) && (taskData->outputs_count[0] > 0) &&
           (taskData->inputs_count[0] == taskData->outputs_count[0]);
  }
  return true;
}

bool mironov_a_broadcast_mpi::ComponentSumPowerBoostImpl::run() {
  internal_order_test();
  internal_order_test();
  // share data
  broadcast(world, delta_, 0);
  broadcast(world, size_, 0);

  result_.resize(size_, 0);
  if (world.rank() != 0) {
    input_.resize(size_);
  }

  broadcast(world, input_.data(), input_.size(), 0);

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); ++proc) {
      world.send(proc, 0, powers_.data() + proc * delta_, delta_);
    }
  }

  std::vector<int> local_powers(delta_);
  if (world.rank() == 0) {
    std::copy(powers_.data(), powers_.data() + delta_, local_powers.begin());
  } else {
    world.recv(0, 0, local_powers.data(), delta_);
  }

  // solve problem
  for (int it = 0; it < static_cast<int>(input_.size()); ++it) {
    int number = input_[it];

    int res = 0;
    for (int power : local_powers) {
      if (power >= 0) {
        res += static_cast<int>(pow(number, power));
      }
    }
    input_[it] = res;
  }

  // share res back to root
  reduce(world, input_.data(), input_.size(), result_.data(), std::plus<int>(), 0);

  return true;
}

bool mironov_a_broadcast_mpi::ComponentSumPowerBoostImpl::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int* answer = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(result_.begin(), result_.end(), answer);
  }
  return true;
}
