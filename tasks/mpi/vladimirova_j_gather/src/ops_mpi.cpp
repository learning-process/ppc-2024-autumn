// Copyright 2023 Nesterov Alexander
#include "mpi/vladimirova_j_gather/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

std::vector<int> vladimirova_j_gather_mpi::noDeadEnds(std::vector<int> way) {
  int i = 0;
  size_t j = i + 1;

  while ((i >= 0) && (j < way.size())) {
    if ((way[i] == -1) && (way[i] == way[j])) {
      do {
        i -= 1;
        j += 1;
        if ((i < 0) || (!(j < way.size()))) {
          break;
        };

        if (((way[i] * way[i] == 1) && (way[i] == (-1) * way[j])) ||
            (way[i] * way[j] == 4)) {  // if rl lr or uu dd   1-1 -11 or 22 -2-2
          way[i] = 0;
          way[j] = 0;

        } else {
          break;
        }

      } while ((i > 0) && (j < way.size()));
      i = j - 1;
    }

    j++;
    i++;
  }

  std::vector<int> ans = std::vector<int>();
  for (auto k : way)
    if (k != 0) ans.push_back(k);
  // way.erase(std::remove(way.begin(), way.end(), 0), way.end());

  return ans;
}

std::vector<int> vladimirova_j_gather_mpi::convertToBinaryTreeOrder(const std::vector<int>& arr) {
  std::vector<int> result;
  result.reserve(arr.size());
  std::vector<int> stack;
  result.reserve(stack.size());
  stack.push_back(0);

  while (!stack.empty()) {
    int rank = stack.back();
    stack.pop_back();

    if ((size_t)rank < arr.size()) {
      result.push_back(arr[rank]);

      int child1 = 2 * rank + 2;
      int child0 = 2 * rank + 1;

      if ((size_t)child1 < arr.size()) stack.push_back(child1);
      if ((size_t)child0 < arr.size()) stack.push_back(child0);
    }
  }

  return result;
}

bool vladimirova_j_gather_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  return true;
}

bool vladimirova_j_gather_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs_count[0] <= 0) return false;
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  // for (size_t i = 0; i < taskData->inputs_count[0]; i++)  std::cout << tmp_ptr[i] << " ";
  // std::cout<< std::endl;
  for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
    switch (tmp_ptr[i]) {
      case 1:
      case 2:
      case -1:
        break;
      default:
        return false;
    }
  }
  return true;
}

bool vladimirova_j_gather_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  res = vladimirova_j_gather_mpi::noDeadEnds(input_);
  return true;
}

bool vladimirova_j_gather_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  taskData->outputs_count[0] = res.size();

  auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(res.begin(), res.end(), output_data);
  // reinterpret_cast<int*>(taskData->outputs[0]) = res;
  return true;
}

bool vladimirova_j_gather_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    // Init vectors
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
  }
  // Init value for output
  return true;
}

bool vladimirova_j_gather_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs_count[0] <= 0) return false;
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
      switch (tmp_ptr[i]) {
        case 1:
        case 2:
        case -1:
          break;
        default:
          return false;
      }
    }
  }
  return true;
}

template <typename T>
bool myGather(std::vector<T>& send_data, int send_count, boost::mpi::communicator world) {
  int rank = world.rank();
  int parent = (rank - 1) / 2;
  int child0 = (2 * rank) + 1;
  int child1 = (2 * rank) + 2;
  if (child0 >= world.size()) child0 = -1;
  if (child1 >= world.size()) child1 = -1;
  std::vector<T> recv_data;
  std::vector<T> child0_data;
  std::vector<T> child1_data;

  if (child0 > 0) {
    int size;
    world.recv(child0, 0, size);
    child0_data = std::vector<T>(size);
    world.recv(child0, 0, child0_data.data(), size);
  }
  if (child1 > 0) {
    int size;
    world.recv(child1, 0, size);
    child1_data = std::vector<T>(size);
    world.recv(child1, 0, child1_data.data(), size);
  }
  recv_data = std::vector<T>(send_count + child0_data.size() + child1_data.size());

  if (rank != 0) {
    std::copy(send_data.begin(), send_data.end(), recv_data.begin());
    std::copy(child0_data.begin(), child0_data.end(), recv_data.begin() + send_count);
    std::copy(child1_data.begin(), child1_data.end(), recv_data.begin() + child0_data.size() + send_count);
    int size = recv_data.size();
    world.send(parent, 0, size);
    world.send(parent, 0, recv_data.data(), recv_data.size());
  }

  if (rank == 0) {
    send_data.insert(send_data.end(), child0_data.begin(), child0_data.end());
    send_data.insert(send_data.end(), child1_data.begin(), child1_data.end());
  }
  return true;
}

bool vladimirova_j_gather_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int rank = world.rank();
  int size;

  if (rank == 0) {
    size = input_.size() / world.size();
    for (int i = 1; i < world.size(); i++) {
      world.send(i, 0, size);
    }

    std::vector<int> pr(world.size());
    for (size_t i = 0; i < pr.size(); i++) {
      pr[i] = i;
    }

    pr = convertToBinaryTreeOrder(pr);

    for (int i = 1; i < world.size(); i++) {
      world.send(pr[i], 0, input_.data() + size * i, size);
    }

    local_input_ = std::vector<int>(input_.begin(), input_.begin() + size);

  } else {
    world.recv(0, 0, size);
    local_input_ = std::vector<int>(size);
    world.recv(0, 0, local_input_.data(), size);
  }

  local_input_ = vladimirova_j_gather_mpi::noDeadEnds(local_input_);

  myGather(local_input_, local_input_.size(), world);

  if (rank == 0) {
    local_input_.insert(local_input_.end(), input_.end() - input_.size() % world.size(), input_.end());
    local_input_ = vladimirova_j_gather_mpi::noDeadEnds(local_input_);
  }

  return true;
}

bool vladimirova_j_gather_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    taskData->outputs_count[0] = local_input_.size();

    auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(local_input_.begin(), local_input_.end(), output_data);

    // reinterpret_cast<int*>(taskData->outputs[0]) = res;
  }
  return true;
}
