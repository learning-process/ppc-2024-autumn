#include "mpi/agafeev_s_linear_topology/include/lintop_mpi.hpp"

namespace agafeev_s_linear_topology {

std::vector<int> calculating_Route(int a, int b) {
  std::vector<int> vec;
  if (a < b)
    for (int i = a; i <= b; i++) vec.push_back(i);
  else
    for (int i = a; i >= b; i--) vec.push_back(i);

  return vec;
}

bool LinearTopology::validation() {
  internal_order_test();

  sender_ = *reinterpret_cast<int*>(taskData->inputs[0]);
  receiver_ = *reinterpret_cast<int*>(taskData->inputs[1]);

  return ((sender_ >= 0) && (receiver_ >= 0) && (sender_ < world.size()) && (receiver_ < world.size()));
}

bool LinearTopology::pre_processing() {
  internal_order_test();

  if (world.rank() == sender_) {
    auto* temp_ptr = reinterpret_cast<int*>(taskData->inputs[2]);
    perfect_way_.insert(perfect_way_.begin(), temp_ptr, temp_ptr + taskData->inputs_count[0]);
  }

  return true;
}

bool LinearTopology::run() {
  internal_order_test();

  if ((sender_ < receiver_ && (world.rank() < sender_ || world.rank() > receiver_))) {
    return true;
  }

  if (sender_ == receiver_) {
    result_ = true;
    return true;
  }

  int mini = std::min(sender_, receiver_);
  int maxi = std::max(sender_, receiver_);

  if (world.rank() <= maxi && world.rank() >= mini) {
    int route_value = (receiver_ - sender_) / abs(receiver_ - sender_);
    if (world.rank() == sender_) {
      ranks_vec_.push_back(world.rank());
      world.send(world.rank() + route_value, 0, ranks_vec_);
      world.send(world.rank() + route_value, 1, perfect_way_);
    } else {
      world.recv(world.rank() - route_value, 0, ranks_vec_);
      world.recv(world.rank() - route_value, 1, perfect_way_);
      ranks_vec_.push_back(world.rank());
      if (world.rank() != receiver_) {
        world.send(world.rank() + route_value, 0, ranks_vec_);
        world.send(world.rank() + route_value, 1, perfect_way_);
      } else if (ranks_vec_ == perfect_way_)
        result_ = true;
    }
  }

  return true;
}

bool LinearTopology::post_processing() {
  internal_order_test();

  if (world.rank() == receiver_) {
    bool* output_data_ptr = reinterpret_cast<bool*>(taskData->outputs[0]);
    output_data_ptr[0] = result_;
  }

  return true;
}

}  // namespace agafeev_s_linear_topology
