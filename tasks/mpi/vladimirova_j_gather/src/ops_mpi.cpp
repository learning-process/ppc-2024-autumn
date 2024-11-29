// Copyright 2023 Nesterov Alexander
#include "mpi/vladimirova_j_gather/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> vladimirova_j_gather_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = (gen() % 2 + 1) * (gen() % 2 - 1);
  }
  return vec;
}

std::vector<int> vladimirova_j_gather_mpi::noDeadEnds(std::vector<int> way) {
  int i = 0;
  size_t j = 1;

  while (j <= way.size()) {
    if ((way[i] * way[i] == 1) && (way[i] == way[j])) {
      do {
        i -= 1;
        j += 1;
        if (!(((size_t)i >= 0) && (j < way.size()))) break;
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

  way.erase(std::remove(way.begin(), way.end(), 0), way.end());
  return way;
}

std::vector<int> vladimirova_j_gather_mpi::noStrangeSteps(std::vector<int> way) {
  std::cout << "NO STR STEPS\n";
  for (int value : way) {
    std::cout << value << " ";
  }

  for (size_t i = 1; i < way.size(); i++)
    if (way[i] == -way[i - 1]) {
      way[i] = 0;
      way[i - 1] = 0;
    }
  way.erase(std::remove(way.begin(), way.end(), 0), way.end());

  for (size_t i = 3; i < way.size(); i++) {
    if (((way[i] == -1) || (way[i] == 1)) && (way[i] == way[i - 1]) && (way[i] == way[i - 2]) &&
        (way[i] == way[i - 3])) {
      way[i] = 0;
      way[i - 1] = 0;
      way[i - 2] = 0;
      way[i - 3] = 0;
    }
  }

  way.erase(std::remove(way.begin(), way.end(), 0), way.end());
  std::cout << "NO STR STEPS\n";
  for (int value : way) {
    std::cout << value << " ";
  }
  std::cout << std::endl;
  return way;
}

std::vector<int> vladimirova_j_gather_mpi::convertToBinaryTreeOrder(const std::vector<int>& arr) {
  std::vector<int> result;
  result.reserve(arr.size());
  std::vector<int> stack;
  result.reserve(stack.size());
  stack.push_back(0);

  while (!stack.empty()) {
    int r = stack.back();
    stack.pop_back();

    if ((size_t)r < arr.size()) {
      result.push_back(arr[r]);

      int child1 = 2 * r + 2;
      int child0 = 2 * r + 1;

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
  return taskData->outputs_count[0] == 1;
}

bool vladimirova_j_gather_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  input_ = vladimirova_j_gather_mpi::noDeadEnds(input_);
  res = vladimirova_j_gather_mpi::noStrangeSteps(input_);
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
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

template <typename T>
bool myGather(std::vector<T>& send_data, int send_count, boost::mpi::communicator world) {
  int r = world.rank();
  int parent = (r - 1) / 2;
  int child0 = (2 * r) + 1;
  int child1 = (2 * r) + 2;
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

  if (r != 0) {
    std::copy(send_data.begin(), send_data.end(), recv_data.begin());
    std::copy(child0_data.begin(), child0_data.end(), recv_data.begin() + send_count);
    std::copy(child1_data.begin(), child1_data.end(), recv_data.begin() + child0_data.size() + send_count);
    int size = recv_data.size();
    world.send(parent, 0, size);
    world.send(parent, 0, recv_data.data(), recv_data.size());
  }

  std::cout << "\nTHERE proc " << r << "\n";
  for (int value : recv_data) {
    std::cout << value << " ";
  }

  std::cout << std::endl;

  if (r == 0) {
    send_data.insert(send_data.end(), child0_data.begin(), child0_data.end());
    send_data.insert(send_data.end(), child1_data.begin(), child1_data.end());

    std::cout << "\nTHERE end 1 " << r << std::endl;
    for (int value : send_data) {
      std::cout << value << " ";
    }

    std::cout << std::endl;
  }
  return true;
}

bool vladimirova_j_gather_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int r = world.rank();
  int size;

  if (r == 0) {
    size = input_.size() / world.size();
    for (size_t i = 1; i < world.size(); i++) {
      world.send(i, 0, size);
    }

    std::vector<int> pr(world.size());
    for (size_t i = 0; i < pr.size(); i++) {
      pr[i] = i;
    }

    std::cout << "TREE" << std::endl;
    std::for_each(pr.begin(), pr.end(), [](int number) { std::cout << number << " "; });
    std::cout << std::endl;

    pr = convertToBinaryTreeOrder(pr);

    std::cout << "TREE" << std::endl;
    std::for_each(pr.begin(), pr.end(), [](int number) { std::cout << number << " "; });
    std::cout << std::endl;

    for (size_t i = 1; i < world.size(); i++) {
      world.send(pr[i], 0, input_.data() + size * i, size);
    }

    local_input_ = std::vector<int>(input_.begin(), input_.begin() + size);

  } else {
    world.recv(0, 0, size);
    local_input_ = std::vector<int>(size);
    world.recv(0, 0, local_input_.data(), size);
  }

  std::cout << "TREE  " << r << "   \n";
  std::for_each(local_input_.begin(), local_input_.end(), [](int number) { std::cout << number << " "; });
  std::cout << std::endl;

  local_input_ = vladimirova_j_gather_mpi::noDeadEnds(local_input_);

  myGather(local_input_, local_input_.size(), world);

  if (r == 0) {
    local_input_.insert(local_input_.end(), input_.end() - input_.size() % world.size(), input_.end());
    std::cout << "ANS  1" << r << "   \n";
    std::for_each(local_input_.begin(), local_input_.end(), [](int number) { std::cout << number << " "; });
    std::cout << std::endl;
    local_input_ = vladimirova_j_gather_mpi::noDeadEnds(local_input_);
    local_input_ = vladimirova_j_gather_mpi::noStrangeSteps(local_input_);
    std::cout << "ANS  2" << r << "   \n";
    std::for_each(local_input_.begin(), local_input_.end(), [](int number) { std::cout << number << " "; });
    std::cout << std::endl;
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
