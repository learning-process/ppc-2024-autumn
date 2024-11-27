#include "mpi/kovalchuk_a_horizontal_tape_scheme/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <random>
#include <string>
#include <vector>

bool kovalchuk_a_horizontal_tape_scheme_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init matrix and vector
  if (taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0) {
    matrix_ = std::vector<std::vector<int>>(taskData->inputs_count[0], std::vector<int>(taskData->inputs_count[1]));
    for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
      auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
      std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[1], matrix_[i].begin());
    }
    vector_ = std::vector<int>(taskData->inputs_count[1]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[taskData->inputs_count[0]]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[1], vector_.begin());
  } else {
    matrix_ = std::vector<std::vector<int>>();
    vector_ = std::vector<int>();
  }
  // Init result vector
  result_ = std::vector<int>(taskData->inputs_count[0], 0);
  return true;
}

bool kovalchuk_a_horizontal_tape_scheme_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool kovalchuk_a_horizontal_tape_scheme_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  if (!matrix_.empty() && !vector_.empty()) {
    for (unsigned int i = 0; i < matrix_.size(); i++) {
      result_[i] = std::inner_product(matrix_[i].begin(), matrix_[i].end(), vector_.begin(), 0);
    }
  }
  return true;
}

bool kovalchuk_a_horizontal_tape_scheme_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  std::copy(result_.begin(), result_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  return true;
}

bool kovalchuk_a_horizontal_tape_scheme_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;

  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] > 0 ? std::max(1u, taskData->inputs_count[0] / world.size()) : 0;
  }
  boost::mpi::broadcast(world, delta, 0);

  local_matrix_rows_.resize(delta * taskData->inputs_count[1]);
  vector_.resize(taskData->inputs_count[1]);

  if (world.rank() == 0) {
    std::vector<int> matrix_flat(taskData->inputs_count[0] * taskData->inputs_count[1]);
    for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
      auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
      std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[1], matrix_flat.begin() + i * taskData->inputs_count[1]);
    }

    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[taskData->inputs_count[0]]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[1], vector_.begin());

    for (int proc = 1; proc < world.size(); proc++) {
      int* buffer = matrix_flat.data() + proc * delta * taskData->inputs_count[1];
      int buffer_size = delta * taskData->inputs_count[1];
      world.send(proc, 0, buffer, buffer_size);
    }

    std::copy(matrix_flat.begin(), matrix_flat.begin() + delta * taskData->inputs_count[1], local_matrix_rows_.begin());
  } else {
    world.recv(0, 0, local_matrix_rows_.data(), delta * taskData->inputs_count[1]);
    boost::mpi::broadcast(world, vector_, 0);
  }

  local_result_.resize(delta, 0);
  return true;
}

bool kovalchuk_a_horizontal_tape_scheme_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == taskData->inputs_count[0];
  }
  return true;
}

bool kovalchuk_a_horizontal_tape_scheme_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  if (!local_matrix_rows_.empty() && !vector_.empty()) {
    for (unsigned int i = 0; i < local_result_.size(); i++) {
      local_result_[i] = std::inner_product(local_matrix_rows_.begin() + i * vector_.size(),
                                            local_matrix_rows_.begin() + (i + 1) * vector_.size(), vector_.begin(), 0);
    }
  }

  std::vector<int> global_result(taskData->inputs_count[0], 0);
  boost::mpi::reduce(world, local_result_.data(), local_result_.size(), global_result.data(), std::plus<int>(), 0);

  if (world.rank() == 0) {
    std::copy(global_result.begin(), global_result.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }

  return true;
}

bool kovalchuk_a_horizontal_tape_scheme_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  return true;
}