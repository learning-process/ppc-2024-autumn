#include "mpi/dormidontov_e_circle_topology/include/dect_ops_mpi.hpp"

#include <algorithm>
#include <climits>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

//----------------------------------------------------------------------------
bool dormidontov_e_circle_topology_mpi::topology::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    size = taskData->inputs_count[0];
    input_.resize(size);
    std::copy(reinterpret_cast<int*>(taskData->inputs[0]), reinterpret_cast<int*>(taskData->inputs[0]) + size,
              input_.begin());
  }
  return true;
}

bool dormidontov_e_circle_topology_mpi::topology::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (((taskData->inputs_count.size() == 1 && taskData->inputs_count[0] > 0) &&
             taskData->outputs_count.size() == 2 && taskData->outputs_count[0] == taskData->inputs_count[0]));
  }
  return true;
}

bool dormidontov_e_circle_topology_mpi::topology::run() {
  internal_order_test();
  // if we have only one process, we dont' do anything
  if (world.size() > 1) {
    if (world.rank() == 0) {
      // this proc will send and wait a rec
      marks_.push_back(0);
      world.send(world.rank() + 1, 0, input_);
      world.send(world.rank() + 1, 0, marks_);

      int from = (2 * world.size() - 1) % world.size();
      world.recv(from, 0, output_);
      world.recv(from, 0, marks_);
      marks_.push_back(0);
    } else {
      int from = world.rank() - 1;
      world.recv(from, 0, input_);
      world.recv(from, 0, marks_);
      marks_.push_back(world.rank());

      int to = (world.rank() + 1) % world.size();
      world.send(to, 0, input_);
      world.send(to, 0, marks_);
    }
  } else {
    output_ = input_;
    marks_.push_back(0);
    marks_.push_back(0);
  }
  return true;
}

bool dormidontov_e_circle_topology_mpi::topology::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int* pr_output = reinterpret_cast<int*>(taskData->outputs[0]);
    int* pr_marks = reinterpret_cast<int*>(taskData->outputs[1]);
    for (int i = 0; i < size; i++) {
      pr_output[i] = output_[i];
    }
    for (int i = 0; i < world.size() + 1; i++) {
      pr_marks[i] = marks_[i];
    }
  }
  return true;
}