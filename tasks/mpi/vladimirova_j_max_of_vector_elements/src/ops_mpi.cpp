#include "mpi/vladimirova_j_max_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

int vladimirova_j_max_of_vector_elements_mpi::FindMaxElem(std::vector<int> m) {
  int max_elem = m[0];
  for (int &i : m) {
    if (i > max_elem) {
      max_elem = i;
    }
  }
  return max_elem;
}

bool vladimirova_j_max_of_vector_elements_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  input_ = std::vector<int>(taskData->inputs_count[0] * taskData->inputs_count[1]);

  for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
    auto *input_data = reinterpret_cast<int *>(taskData->inputs[i]);
    for (unsigned int j = 0; j < taskData->inputs_count[1]; j++) {
      input_[i * taskData->inputs_count[1] + j] = input_data[j];
    }
  }
  return true;
}

bool vladimirova_j_max_of_vector_elements_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  return (taskData->outputs_count[0] == 1) && ((taskData->inputs_count[0] > 0) && (taskData->inputs_count[1] > 0));
}

bool vladimirova_j_max_of_vector_elements_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  res = vladimirova_j_max_of_vector_elements_mpi::FindMaxElem(input_);
  return true;
}

bool vladimirova_j_max_of_vector_elements_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  reinterpret_cast<int *>(taskData->outputs[0])[0] = res;
  return true;
}

bool vladimirova_j_max_of_vector_elements_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    unsigned int rows = taskData->inputs_count[0];
    unsigned int columns = taskData->inputs_count[1];

    input_ = std::vector<int>(rows * columns);

    for (unsigned int i = 0; i < rows; i++) {
      auto *input_data = reinterpret_cast<int *>(taskData->inputs[i]);
      for (unsigned int j = 0; j < columns; j++) {
        input_[i * columns + j] = input_data[j];
      }
    }
  }

  return true;
}

bool vladimirova_j_max_of_vector_elements_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  return (world.rank() != 0) || ((taskData->outputs_count[0] == 1) && (!taskData->inputs.empty()));
}

bool vladimirova_j_max_of_vector_elements_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  unsigned int delta = 0;

  if (world.rank() == 0) {
    // Init vectors

    unsigned int rows = taskData->inputs_count[0];
    unsigned int columns = taskData->inputs_count[1];

    delta = columns * rows / world.size();
    int div_r = columns * rows % world.size() + 1;

    if (delta == 0) {
      for (int i = 1; i < world.size(); i++) {
        world.send(i, 0, 0);
      }
      local_input_ = std::vector<int>(input_.begin(), input_.begin() + div_r);
      res = vladimirova_j_max_of_vector_elements_mpi::FindMaxElem(local_input_);
      return true;
    }
    for (int i = 1; i < world.size(); i++) {
      world.send(i, 0, delta + (int)(i < div_r));
    }

    for (int i = 1; i < div_r; i++) {
      world.send(i, 0, input_.data() + delta * i + i - 1, delta + 1);
    }
    for (int i = div_r; i < world.size(); i++) {
      world.send(i, 0, input_.data() + delta * i + div_r - 1, delta);
    }

    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta);
  }

  if (world.rank() != 0) {
    world.recv(0, 0, delta);
    if (delta == 0) return true;
    local_input_ = std::vector<int>(delta);
    world.recv(0, 0, local_input_.data(), delta);
  }

  int local_res = vladimirova_j_max_of_vector_elements_mpi::FindMaxElem(local_input_);
  reduce(world, local_res, res, boost::mpi::maximum<int>(), 0);

  return true;
}

bool vladimirova_j_max_of_vector_elements_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<int *>(taskData->outputs[0])[0] = res;
  }
  return true;
}
