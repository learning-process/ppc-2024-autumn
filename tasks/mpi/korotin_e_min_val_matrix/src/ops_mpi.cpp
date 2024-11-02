#include "mpi/korotin_e_min_val_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<double> korotin_e_min_val_matrix_mpi::getRandomMatrix(const unsigned rows, const unsigned columns,
                                                                  double scal) {
  if (rows == 0 || columns == 0) {
    throw std::invalid_argument("Can't creaate matrix with 0 rows or columns");
  }

  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<double> matrix(rows * columns);
  for (unsigned i = 0; i < rows * columns; i++) {
    matrix[i] = gen() / scal;
  }
  return matrix;
}

bool korotin_e_min_val_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<double>(taskData->inputs_count[0]);
  auto* start = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(start, start + taskData->inputs_count[0], input_.begin());
  res = 0.0;
  return true;
}

bool korotin_e_min_val_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool korotin_e_min_val_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  res = input_[0];
  for (unsigned i = 1; i < taskData->inputs_count[0]; i++) {
    if (input_[i] < res) res = input_[i];
  }
  return true;
}

bool korotin_e_min_val_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}

bool korotin_e_min_val_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_ = std::vector<double>(taskData->inputs_count[0]);
    auto* start = reinterpret_cast<double*>(taskData->inputs[0]);
    std::copy(start, start + taskData->inputs_count[0], input_.begin());
  }
  res = 0.0;
  return true;
}

bool korotin_e_min_val_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool korotin_e_min_val_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  unsigned int delta = 0;
  int remainder = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
    remainder = taskData->inputs_count[0] % world.size();
  }
  broadcast(world, delta, 0);
  broadcast(world, remainder, 0);

  if (world.rank() == 0) {
    int counter = 1;
    for (int proc = 1; proc < world.size(); proc++) {
      if (counter < remainder) {
        world.send(proc, 0, input_.data() + proc * delta + counter, delta + 1);
        counter++;
      } else
        world.send(proc, 0, input_.data() + proc * delta + remainder, delta);
    }
  }

  if (world.rank() < remainder) {
    local_input_ = std::vector<double>(delta + 1);
  } else
    local_input_ = std::vector<double>(delta);

  if (world.rank() == 0) {
    if (remainder > 0)
      local_input_ = std::vector<double>(input_.begin(), input_.begin() + delta + 1);
    else
      local_input_ = std::vector<double>(input_.begin(), input_.begin() + delta);
  } else {
    if (world.rank() < remainder) {
      world.recv(0, 0, local_input_.data(), delta + 1);
    } else
      world.recv(0, 0, local_input_.data(), delta);
  }
  double local_res;

  if (local_input_.empty())
    local_res = INFINITY;
  else {
    local_res = local_input_[0];
    for (std::vector<double>::size_type i = 1; i < local_input_.size(); i++) {
      if (local_input_[i] < local_res) local_res = local_input_[i];
    }
  }

  reduce(world, local_res, res, boost::mpi::minimum<double>(), 0);

  return true;
}

bool korotin_e_min_val_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
