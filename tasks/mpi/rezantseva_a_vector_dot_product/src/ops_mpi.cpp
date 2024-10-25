// Copyright 2024 Nesterov Alexander
#include "mpi/rezantseva_a_vector_dot_product/include/ops_mpi.hpp"

#include <random>
#include <thread>

static int offset = 0;
using namespace std::chrono_literals;
// Sequential
std::vector<int> rezantseva_a_vector_dot_product_mpi::createRandomVector(int v_size) {
  std::vector<int> vec(v_size);
  std::mt19937 gen;
  gen.seed((unsigned)time(nullptr) + ++offset);
  for (int i = 0; i < v_size; i++) vec[i] = gen() % 100;
  return vec;
}

int rezantseva_a_vector_dot_product_mpi::vectorDotProduct(const std::vector<int>& v1, const std::vector<int>& v2) {
  if (v1.size() != v2.size()) throw "Error! Vectors must have equal size!";
  long long result = 0;
  for (size_t i = 0; i < v1.size(); i++) result += v1[i] * v2[i];
  return result;
}

bool rezantseva_a_vector_dot_product_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return (taskData->inputs.size() == taskData->inputs_count.size() && taskData->inputs.size() == 2) &&
         (taskData->inputs_count[0] == taskData->inputs_count[1]) &&
         (taskData->outputs.size() == taskData->outputs_count.size()) && taskData->outputs.size() == 1 &&
         taskData->outputs_count[0] == 1;
}

bool rezantseva_a_vector_dot_product_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output

  input_ = std::vector<std::vector<int>>(taskData->inputs.size());
  for (size_t i = 0; i < input_.size(); i++) {
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
    input_[i] = std::vector<int>(taskData->inputs_count[i]);
    for (size_t j = 0; j < taskData->inputs_count[i]; j++) {
      input_[i][j] = tmp_ptr[j];
    }
  }
  res = 0;
  return true;
}

bool rezantseva_a_vector_dot_product_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < input_[0].size(); i++) {
    res += input_[0][i] * input_[1][i];
  }

  return true;
}

bool rezantseva_a_vector_dot_product_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

// Parallel

bool rezantseva_a_vector_dot_product_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return (taskData->inputs.size() == taskData->inputs_count.size() && taskData->inputs.size() == 2) &&
           (taskData->inputs_count[0] == taskData->inputs_count[1]) &&
           (taskData->outputs.size() == taskData->outputs_count.size()) && taskData->outputs.size() == 1 &&
           taskData->outputs_count[0] == 1;
  }
  return true;
}

bool rezantseva_a_vector_dot_product_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  unsigned int delta = 0;
  unsigned int count_rank_add = 0;
  unsigned int count_rank = 0;
  const int num_processes = world.size();

  if (num_processes == 0) {
    return false;
  }

  if (num_processes > 1) {
    if (world.rank() == 0) {
      delta = taskData->inputs_count[0] / num_processes;
      count_rank_add = taskData->inputs_count[0] % num_processes;
      count_rank = num_processes - count_rank_add;
    }

    broadcast(world, delta, 0);
    broadcast(world, count_rank, 0);
    broadcast(world, count_rank_add, 0);
  }

  if (world.rank() == 0) {
    input_ = std::vector<std::vector<int>>(taskData->inputs.size());
    for (size_t i = 0; i < input_.size(); i++) {
      auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
      input_[i] = std::vector<int>(taskData->inputs_count[0]);
      for (size_t j = 0; j < taskData->inputs_count[0]; j++) {
        input_[i][j] = tmp_ptr[j];
      }
    }
    for (int proc = 1; proc < num_processes; proc++) {
      if (proc < count_rank) {
        world.send(proc, 0, input_[0].data() + proc * delta, delta);
        world.send(proc, 1, input_[1].data() + proc * delta, delta);
      } else {
        int remaining_elements = delta + 1;
        world.send(proc, 0, input_[0].data() + count_rank * delta, remaining_elements);
        world.send(proc, 1, input_[1].data() + count_rank * delta, remaining_elements);
      }
    }
  }

  local_input1_ = std::vector<int>(delta);
  local_input2_ = std::vector<int>(delta);

  if (world.rank() == 0) {
    local_input1_ = std::vector<int>(input_[0].begin(), input_[0].begin() + delta);
    local_input2_ = std::vector<int>(input_[1].begin(), input_[1].begin() + delta);
  } else {
    if (world.rank() < count_rank) {
      world.recv(0, 0, local_input1_.data(), delta);
      world.recv(0, 1, local_input2_.data(), delta);
    } else {
      int remaining_elements = delta + 1;
      local_input1_ = std::vector<int>(remaining_elements);
      local_input2_ = std::vector<int>(remaining_elements);
      world.recv(0, 0, local_input1_.data(), remaining_elements);
      world.recv(0, 1, local_input2_.data(), remaining_elements);
    }
  }
  res = 0;
  return true;
}

bool rezantseva_a_vector_dot_product_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int local_res = 0;

  for (size_t i = 0; i < local_input1_.size(); i++) {
    local_res += local_input1_[i] * local_input2_[i];
  }

  std::vector<int> all_res;
  boost::mpi::gather(world, local_res, all_res, 0);

  if (world.rank() == 0) {
    for (int result : all_res) {
      res += result;
    }
  }
  std::this_thread::sleep_for(20ms);
  return true;
}

bool rezantseva_a_vector_dot_product_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
