
#include "mpi/sharamygina_i_line_topology/include/ops_mpi.h"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

bool sharamygina_i_line_topology_mpi::line_topology_mpi::pre_processing() {
  internal_order_test();

  sendler = taskData->inputs_count[0];
  recipient = taskData->inputs_count[1];
  msize = taskData->inputs_count[2];

  if (world.rank() == sendler) {
    auto* inputBuffer = reinterpret_cast<int*>(taskData->inputs[0]);
    message.assign(inputBuffer, inputBuffer + msize);
  }

  return true;
}

bool sharamygina_i_line_topology_mpi::line_topology_mpi::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() < 3) return false;

  int sendler_ = taskData->inputs_count[0];
  int recipient_ = taskData->inputs_count[1];
  int msize_ = taskData->inputs_count[2];

  return (sendler_ >= 0 && sendler_ < world.size() && recipient_ >= 0 && recipient_ < world.size() && msize_ > 0) &&
         ((world.rank() != sendler_) || ((!taskData->inputs.empty()) && (taskData->inputs[0] != nullptr))) &&
         ((world.rank() != recipient_) || ((!taskData->outputs.empty()) && (taskData->outputs[0] != nullptr)));
}

bool sharamygina_i_line_topology_mpi::line_topology_mpi::run() {
  internal_order_test();

  if (sendler == recipient) {
    return true;
  }

  if (world.rank() < sendler || world.rank() > recipient) {
    return true;
  }
  if (world.rank() == sendler) {
    world.send(world.rank() + 1, 0, message);
  } else {
    world.recv(world.rank() - 1, 0, message);
    if (world.rank() < recipient) {
      world.send(world.rank() + 1, 0, message);
    }
  }
  return true;
}

bool sharamygina_i_line_topology_mpi::line_topology_mpi::post_processing() {
  internal_order_test();

  if (world.rank() == recipient) {
    auto* mptr = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(message.begin(), message.end(), mptr);
  }

  return true;
}