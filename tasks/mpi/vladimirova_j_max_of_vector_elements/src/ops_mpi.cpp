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
  for (size_t i = 0; i < m.size(); i++) {
    if (m[i] > max_elem) {
      max_elem = m[i];
    }
  }
  return max_elem;
}
std::vector<int> vladimirova_j_max_of_vector_elements_mpi::CreateVector(size_t size, size_t spread_of_val) {
  if (size == 0) throw "null size";
  // Init value for input and output
  std::random_device dev;
  std::mt19937 random(dev());
  std::vector<int> v(size);
  for (size_t i = 0; i < size; i++) {
    v[i] = (random() % (2 * spread_of_val + 1)) - spread_of_val;
  }
  return v;
}

std::vector<std::vector<int>> vladimirova_j_max_of_vector_elements_mpi::CreateInputMatrix(size_t row_c, size_t col_c,
                                                                                          size_t spread_of_val) {
  if ((row_c == 0) || (col_c == 0)) throw "null size";
  std::vector<std::vector<int>> m(row_c);
  for (size_t i = 0; i < row_c; i++) {
    m[i] = vladimirova_j_max_of_vector_elements_mpi::CreateVector(col_c, spread_of_val);
  }
  return m;
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

  res = INT_MIN;
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
  //
  unsigned int delta = 0;
  if (world.rank() == 0) {
    // Init vectors

    unsigned int rows = taskData->inputs_count[0];
    unsigned int columns = taskData->inputs_count[1];

    input_ = std::vector<int>(rows * columns);

    for (unsigned int i = 0; i < rows; i++) {
      auto *input_data = reinterpret_cast<int *>(taskData->inputs[i]);
      for (unsigned int j = 0; j < columns; j++) {
        input_[i * columns + j] = input_data[j];
      }
    }

    delta = columns * rows / world.size();
    int div_r = columns * rows % world.size() + 1;

    for (int i = 1; i < world.size(); i++) {
      world.send(i, 0, delta + (int)(i < div_r));
    }

    for (int i = 1; i < div_r; i++) {
      world.send(i, 0, input_.data() + (delta + 1) * i, delta + 1);
    }
    for (int i = div_r; i < world.size(); i++) {
      world.send(i, 0, input_.data() + delta * i + div_r, delta);
    }

    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta);
  }

  if (world.rank() != 0) {
    world.recv(0, 0, delta);
    local_input_ = std::vector<int>(delta);
    world.recv(0, 0, local_input_.data(), delta);
  }

  res = INT_MIN;
  return true;
}

bool vladimirova_j_max_of_vector_elements_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  return (world.rank() != 0) || ((taskData->outputs_count[0] == 1) && (!taskData->inputs.empty()));
}

bool vladimirova_j_max_of_vector_elements_mpi::TestMPITaskParallel::run() {
  internal_order_test();

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
