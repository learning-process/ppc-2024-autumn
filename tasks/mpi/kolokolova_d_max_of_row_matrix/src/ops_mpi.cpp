#include "mpi/kolokolova_d_max_of_row_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> kolokolova_d_max_of_row_matrix_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  std::uniform_int_distribution<int> dist(-100, 99);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}
bool kolokolova_d_max_of_row_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  auto row_count = static_cast<size_t>(*taskData->inputs[1]);
  size_t col_count = taskData->inputs_count[0] / row_count;

  input_.resize(row_count, std::vector<int>(col_count));

  int* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (size_t i = 0; i < row_count; ++i) {
    for (size_t j = 0; j < col_count; ++j) {
      input_[i][j] = input_ptr[i * col_count + j];
    }
  }
  res.resize(row_count);
  return true;
}

bool kolokolova_d_max_of_row_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return *taskData->inputs[1] == taskData->outputs_count[0];
}

bool kolokolova_d_max_of_row_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < input_.size(); ++i) {
    int max_value = input_[i][0];
    for (size_t j = 1; j < input_[i].size(); ++j) {
      if (input_[i][j] > max_value) {
        max_value = input_[i][j];
      }
    }
    res[i] = max_value;
  }
  return true;
}

bool kolokolova_d_max_of_row_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  int* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
  for (size_t i = 0; i < res.size(); ++i) {
    output_ptr[i] = res[i];
  }
  return true;
}

bool kolokolova_d_max_of_row_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  unsigned int delta = 0;
  int proc_rank = world.rank();

  if (proc_rank == 0) {
    delta = taskData->inputs_count[0] / world.size();
  }
  broadcast(world, delta, 0);

  if (proc_rank == 0) {
    // Init vectors
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);

    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * delta, delta);
    }
  }

  local_input_ = std::vector<int>(delta);

  if (proc_rank == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta);
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }
  // Init value for output
  res.resize(world.size());
  return true;
}

bool kolokolova_d_max_of_row_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output and input
    if (taskData->outputs_count[0] == 0 || taskData->inputs_count[0] == 0) return false;
  }
  return true;
}

bool kolokolova_d_max_of_row_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int local_res = 0;
  for (int i = 0; i < int(local_input_.size()); i++) {
    if (local_res < local_input_[i]) local_res = local_input_[i];
  }
  gather(world, local_res, res, 0);
  return true;
}

bool kolokolova_d_max_of_row_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < world.size(); i++) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = res[i];
    }
  }
  return true;
}