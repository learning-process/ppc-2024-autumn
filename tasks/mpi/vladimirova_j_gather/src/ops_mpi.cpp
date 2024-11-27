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
    vec[i] = gen() % 100;
  }
  return vec;
}

std::vector<int> noDeadEnds(std::vector<int> way) {
  
  int i = 0;
   int j = 1;
  int del = 0;
  while (j <= way.size()) {
      if ((way[i] * way[i] == 1) && (way[i] == way[j])) {
          do {
              i -= 1;
              j += 1;

              if (((way[i] * way[i]==1) && (way[i] == (-1) * way[j])) ||
                  (way[i] * way[j] == 4)) {  // if rl lr or uu dd   1-1 -11 or 22 -2-2
                way[i] = 0;
                way[j] = 0;
                del += 2;
              } else {
                break;
              }

          } while ((i > 0) && (j<way.size()));
          i = j - 1;
      }
      j++;
      i++;
  }

  way.erase(std::remove(way.begin(), way.end(), 0), way.end());
  return way;
}

bool vladimirova_j_gather_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  res = 0;
  return true;
}

bool vladimirova_j_gather_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1;
}

bool vladimirova_j_gather_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  return true;
}

bool vladimirova_j_gather_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool vladimirova_j_gather_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    // Init vectors
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
  }
  std::for_each(input_.begin(), input_.end(), [](int number) { std::cout << number << " "; });
  std::cout << " AAAAAAAAAAAAA \n";
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

  std::cout << "\nTHERE proc " << r <<"\n";
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

std::vector<int> convertToBinaryTreeOrder(const std::vector<int>& arr) {
  std::vector<int> result;
  result.reserve(arr.size());
  std::vector<int> stack;
  result.reserve(stack.size());
  stack.push_back(0);


  while (!stack.empty()) {
    int r = stack.back();
    stack.pop_back();

    if (r < arr.size()) {
      result.push_back(arr[r]);

      int child1 = 2 * r + 2;
      int child0 = 2 * r + 1;

      if (child1 < arr.size()) stack.push_back(child1);
      if (child0 < arr.size()) stack.push_back(child0);
    }
  }

  return result;
}

bool vladimirova_j_gather_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int r = world.rank();
  int size;
  /*
  if (r==0) {
    size = input_.size() / world.size();
    for (int i = 1; i < world.size(); i++) {
      world.send(i, 0 ,size);
      world.send(i, 0, input_.data() + size * i, size);
    }
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + size );
  }
  else {
    world.recv(0, 0, size);
    local_input_ = std::vector<int>(size);
    world.recv(0, 0, local_input_.data(), size);
  }
  */

  if (r == 0) {
    size = input_.size() / world.size();
    for (int i = 1; i < world.size(); i++) {
      world.send(i, 0, size);
    }

    std::vector<int> pr(world.size());
    for (int i = 0; i < pr.size(); i++) {
      pr[i] = i;
    }

    std::cout << "TREE" << std::endl;
    std::for_each(pr.begin(), pr.end(), [](int number) { std::cout << number << " "; });
    std::cout << std::endl;

    pr = convertToBinaryTreeOrder(pr);

    std::cout << "TREE" << std::endl;
    std::for_each(pr.begin(), pr.end(), [](int number) { std::cout << number << " "; });
    std::cout << std::endl;

    for (int i = 1; i < world.size(); i++) {
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

  local_input_ = noDeadEnds(local_input_);

  myGather(local_input_, local_input_.size(), world);

  if (r == 0) {
      local_input_.insert(local_input_.end(), input_.end() - input_.size()%world.size(), input_.end());
      std::cout << "ANS  1" << r << "   \n";
      std::for_each(local_input_.begin(), local_input_.end(), [](int number) { std::cout << number << " "; });
      std::cout << std::endl;
      local_input_ = noDeadEnds(local_input_);

      std::cout << "ANS  2" << r << "   \n";
      std::for_each(local_input_.begin(), local_input_.end(), [](int number) { std::cout << number << " "; });
      std::cout << std::endl;
  }
  /*
  local



  int child0_i_s = (((data_size - size) / size) / ((int)(child0 > 0) + (int)(child1 > 0))) * size;

  if (child0 >= world.size()) child0 = -1;
  else {
    world.send(child0, 0, data_size - size);
    world.send(child0, 0, tmp.data() + size,  child0_i_s);
  }
  if (child1 >= world.size()) child1 = -1;
  else {
    world.send(child0, 0, data_size - size);
    world.send(child0, 0, tmp.data() + size + child0_i_s, data_size - size - child0_i_s);
  }

  std::vector<int> myres = noDeadEnds(local_input_);

  std::cout << " PPPPPPPPP  \n";
  std::for_each(myres.begin(), myres.end(), [](int number) { std::cout << number << " "; });
  std::cout << "\nTHERE\n";

  if (child0 > 0) {
    world.recv(child0, 0, child0_res_size);
    //std::cout << child0 + " give " + child0_res;
  }
  if (child1 > 0) {
    world.recv(child1, 0, child1_res_size);
  }

  std::vector<int> local_res = std::vector<int>(myres.size() + child1_res_size + child0_res_size);

  if (child0 > 0) {
    world.recv(child0, 0, local_res.data() + myres.size(), child0_res_size);
    // std::cout << child0 + " give " + child0_res;
  }
  if (child1 > 0) {
    child1_res = std::vector<int>(child1_res_size);
    world.recv(child1, 0, local_res.data() + myres.size() + child0_res_size , child1_res_size);
  }

  //std::cout << " PROC " << world.rank() << " left " << child0 << " right " << child1 << " parent "<<
  parent<<std::endl;

  if (world.rank() == 0) {
    res = local_res;
    return true;
   // std::cout << res << " ansver\n";
  }
  world.send(parent, 0, local_res.size());
  world.send(parent, 0, local_res.data(), local_res.size());
  */

  return true;
}

bool vladimirova_j_gather_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
