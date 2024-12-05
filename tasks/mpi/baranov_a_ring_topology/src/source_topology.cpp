#include "mpi/baranov_a_ring_topology/include/header_topology.hpp"

namespace baranov_a_ring_topology_mpi {
template <class iotype>
bool ring_topology<iotype>::pre_processing() {
  internal_order_test();
  int myid = world.rank();

  int n;
  if (myid == 0) {
    n = taskData->inputs_count[0];
    input_ = std::vector<iotype>(n);
    output_ = std::vector<iotype>(n);
    void* ptr_r = taskData->inputs[0];
    void* ptr_d = input_.data();
    memcpy(ptr_d, ptr_r, sizeof(iotype) * n);
    vec_size_ = n;
  }
  return true;
}
template <class iotype>
bool ring_topology<iotype>::run() {
  internal_order_test();
  boost::mpi::broadcast(world, vec_size_, 0);
  int my_rank = world.rank();
  int sz = world.size();
  if (world.size() != 1) {
    if (my_rank == 0) {
      poll_.push_back(0);
      world.send(my_rank + 1, 0, input_);
      world.send(my_rank + 1, 0, poll_);
    } else {
      std::vector<iotype> buff(vec_size_);
      int tmp_recv = my_rank - 1;
      int tmp_send = (my_rank + 1) % sz;

      world.recv(tmp_recv, 0, buff);
      world.recv(tmp_recv, 0, poll_);

      poll_.push_back(my_rank);

      world.send(tmp_send, 0, buff);
      world.send(tmp_send, 0, poll_);
    }
    if (my_rank == 0) {
      world.recv(world.size() - 1, 0, output_);
      poll_.resize(sz);
      world.recv(world.size() - 1, 0, poll_);
    }
  } else {
    poll_.push_back(0);
    void* ptr_r = output_.data();
    void* ptr_d = input_.data();
    memcpy(ptr_r, ptr_d, sizeof(iotype) * vec_size_);
  }

  return true;
}
template <class iotype>
bool ring_topology<iotype>::post_processing() {
  internal_order_test();
  world.barrier();
  if (world.rank() == 0) {
    for (int i = 0; i != vec_size_; ++i) {
      reinterpret_cast<iotype*>(taskData->outputs[0])[i] = output_[i];
    }
    auto tmp = world.size();
    for (int i = 0; i != tmp; ++i) {
      reinterpret_cast<int*>(taskData->outputs[1])[i] = poll_[i];
    }

    return true;
  }
  return true;
}
template <class iotype>
bool ring_topology<iotype>::validation() {
  internal_order_test();
  // Check count elements of output
  if (world.rank() == 0) {
    if (taskData->outputs_count[0] == 1 && taskData->inputs_count.size() == 1 && taskData->inputs_count[0] >= 0) {
      return true;
    }
  }
  return true;
}

template class baranov_a_ring_topology_mpi::ring_topology<int>;

template class baranov_a_ring_topology_mpi::ring_topology<double>;

template class baranov_a_ring_topology_mpi::ring_topology<unsigned>;
}  // namespace baranov_a_ring_topology_mpi