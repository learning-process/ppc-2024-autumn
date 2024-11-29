#include "mpi/grudzin_k_all_reduce_student/include/ops_mpi.hpp"

#include <algorithm>
#include <climits>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

bool grudzin_k_all_reduce_student_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  rows = taskData->inputs_count[0];
  colums = taskData->inputs_count[1];

  input_.resize(rows * colums);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < colums; ++j) {
      input_[i * colums + j] = reinterpret_cast<int*>(taskData->inputs[0])[i * (colums) + j];
    }
  }

  res_.resize(colums, INT_MAX);
  cnt_.resize(colums, 0);
  return true;
}

bool grudzin_k_all_reduce_student_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return ((taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0) &&
          taskData->inputs_count[1] == taskData->outputs_count[0]);
}

bool grudzin_k_all_reduce_student_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < colums; j++) {
      res_[j] = std::min(res_[j], input_[i * colums + j]);
    }
  }
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < colums; j++) {
      if (input_[i * colums + j] == res_[j]) cnt_[j]++;
    }
  }
  return true;
}

bool grudzin_k_all_reduce_student_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < colums; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = cnt_[i];
  }
  return true;
}

//----------------------------------------------------------------------------
bool grudzin_k_all_reduce_student_mpi::TestMPITaskMyRealization::pre_processing() {
  internal_order_test();
  return true;
}

bool grudzin_k_all_reduce_student_mpi::TestMPITaskMyRealization::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (((taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0) &&
             taskData->inputs_count[1] == taskData->outputs_count[0]));
  }
  return true;
}

template <typename T>
void grudzin_k_all_reduce_student_mpi::TestMPITaskMyRealization::my_all_reduce(const boost::mpi::communicator& world,
                                                                               const T* in_values, int n,
                                                                               T* out_values) {
  int root = world.rank();
  T* cv1 = nullptr;
  T* cv2 = nullptr;
  // Initialisation (out_values=in_values)
  for (int i = 0; i < n; ++i) out_values[i] = in_values[i];
  if (2 * root + 1 < world.size()) {
    // Wait data from our childrens
    cv1 = new int[n];
    world.recv(2 * root + 1, 0, cv1, n);
  }
  if (2 * root + 2 < world.size()) {
    cv2 = new int[n];
    world.recv(2 * root + 2, 0, cv2, n);
  }
  // if we get before calculate we will lose some memory, but we do two cycles in one and win some time

  // calculation
  if (cv1 != nullptr && cv2 != nullptr) {
    for (int i = 0; i < n; ++i) out_values[i] = std::min({out_values[i], cv1[i], cv2[i]});
  } else if (cv1 != nullptr) {
    for (int i = 0; i < n; ++i) out_values[i] = std::min(out_values[i], cv1[i]);
  }
  // delete cv1 and cv2
  delete[] cv1;
  delete[] cv2;
  // send data to our parent (if it exist) and get out_values from him
  if (root > 0) {
    int local_root = root - 1;
    local_root >>= 1;
    world.send(local_root, 0, out_values, n);
    // wait for broadcast
    world.recv(local_root, 0, out_values, n);
  }
  // send to children's
  if (2 * root + 1 < world.size()) {
    world.send(2 * root + 1, 0, out_values, n);
  }
  if (2 * root + 2 < world.size()) {
    world.send(2 * root + 2, 0, out_values, n);
  }
}

bool grudzin_k_all_reduce_student_mpi::TestMPITaskMyRealization::run() {
  internal_order_test();
  int size;
  int rest;
  int delta;
  if (world.rank() == 0) {
    rows = taskData->inputs_count[0];
    colums = taskData->inputs_count[1];
    size = rows * colums;
    rest = size % world.size();
    delta = size / world.size();
  }
  boost::mpi::broadcast(world, size, 0);
  boost::mpi::broadcast(world, rest, 0);
  boost::mpi::broadcast(world, delta, 0);
  boost::mpi::broadcast(world, colums, 0);

  if (rest != 0) {
    delta++;
  }

  if (world.rank() == 0) {
    input_.assign(delta * world.size(), INT_MAX);
    std::copy(reinterpret_cast<int*>(taskData->inputs[0]), reinterpret_cast<int*>(taskData->inputs[0]) + size,
              input_.begin());
  }

  std::vector<int> locmin(colums, INT_MAX);
  local_input_.resize(delta);

  int start = delta;
  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + start, delta);
      start += delta;
    }
  }

  start = delta * world.rank();
  if (world.rank() == 0) {
    std::copy(input_.begin(), input_.begin() + delta, local_input_.begin());
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }

  for (size_t i = 0; i < local_input_.size(); ++i) {
    if (locmin[(start + i) % colums] > local_input_[i]) {
      locmin[(start + i) % colums] = local_input_[i];
    }
  }
  res_.resize(colums, INT_MAX);
  my_all_reduce(world, locmin.data(), colums, res_.data());
  std::vector<int> local_cnt_(colums, 0);
  for (size_t i = 0; i < local_input_.size(); ++i) {
    if (res_[(start + i) % colums] == local_input_[i]) {
      local_cnt_[(start + i) % colums]++;
    }
  }
  cnt_.resize(colums, 0);
  boost::mpi::reduce(world, local_cnt_.data(), colums, cnt_.data(), std::plus<>(), 0);
  return true;
}

bool grudzin_k_all_reduce_student_mpi::TestMPITaskMyRealization::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < colums; i++) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = cnt_[i];
    }
  }
  return true;
}