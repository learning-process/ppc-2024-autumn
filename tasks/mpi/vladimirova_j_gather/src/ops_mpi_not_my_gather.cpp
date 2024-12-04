// Copyright 2023 Nesterov Alexander
#include "mpi/vladimirova_j_gather/include/ops_mpi_not_my_gather.hpp"

#include <algorithm>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "mpi/vladimirova_j_gather/include/ops_mpi.hpp"

using namespace std::chrono_literals;
using namespace vladimirova_j_gather_mpi;

bool vladimirova_j_not_my_gather_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    // Init vectors
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
  }
  // Init value for output
  return true;
}

bool vladimirova_j_not_my_gather_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool vladimirova_j_not_my_gather_mpi::TestMPITaskParallel::run() {
  internal_order_test();


  int r = world.rank();
  int size;
  std::vector<std::vector<int>> root_vec;

  if (r == 0) {
    root_vec = std::vector<std::vector<int>>(world.size());
    size = input_.size() / world.size();

    for (int i = 1; i < world.size(); i++) {
      root_vec[i] = std::vector<int>(size);
    }

    for (int i = 1; i < world.size(); i++) {
      world.send(i, 0, size);
      world.send(i, 0, input_.data() + size * i, size);
    }

    local_input_ = std::vector<int>(input_.begin(), input_.begin() + size);

  } else {
    world.recv(0, 0, size);
    local_input_ = std::vector<int>(size);
    world.recv(0, 0, local_input_.data(), size);
  }


  gather(world, local_input_, root_vec, 0);

  if (r == 0) {
    for (int i = 1; i < world.size(); i++) {
      local_input_.insert(local_input_.end(), root_vec[i].begin(), root_vec[i].end());
    }
    local_input_.insert(local_input_.end(), input_.end() - input_.size() % world.size(), input_.end());

    std::cout << "ANS  1" << r << "   \n";

    std::for_each(local_input_.begin(), local_input_.end(), [](int number) { std::cout << number << " "; });
    std::cout << std::endl;
    local_input_ = vladimirova_j_gather_mpi::noDeadEnds(local_input_);
    local_input_ = vladimirova_j_gather_mpi::noStrangeSteps(local_input_);

    std::cout << "ANS  2 " << r << "   \n";
    std::for_each(local_input_.begin(), local_input_.end(), [](int number) { std::cout << number << " "; });
    std::cout << std::endl;
  }

  return true;
}

bool vladimirova_j_not_my_gather_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    taskData->outputs_count[0] = local_input_.size();
    auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(local_input_.begin(), local_input_.end(), output_data);

    // reinterpret_cast<int*>(taskData->outputs[0]) = res;
  }
  return true;
}
