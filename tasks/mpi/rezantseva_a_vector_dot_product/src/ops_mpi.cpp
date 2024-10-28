// Copyright 2024 Nesterov Alexander
#include "mpi/rezantseva_a_vector_dot_product/include/ops_mpi.hpp"

int rezantseva_a_vector_dot_product_mpi::vectorDotProduct(const std::vector<int>& v1, const std::vector<int>& v2) {
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

  size_t total_elements = 0;
  size_t num_processes = 0;
  size_t delta = 0;
  size_t remainder = 0;

  if (world.rank() == 0) {
    total_elements = taskData->inputs_count[0];
    num_processes = world.size();
    delta = total_elements / num_processes;      // Calculate base size for each process
    remainder = total_elements % num_processes;  // Calculate remaining elements
  }
  boost::mpi::broadcast(world, num_processes, 0);
  boost::mpi::broadcast(world, delta, 0);
  boost::mpi::broadcast(world, remainder, 0);

  std::vector<unsigned int> counts(num_processes);  // Vector to store counts for each process

  if (world.rank() == 0) {
    // Distribute sizes to each process
    for (unsigned int i = 0; i < num_processes; ++i) {
      counts[i] = delta + (i < remainder ? 1 : 0);  // Assign 1 additional element to the first 'remainder' processes
    }
  }
  boost::mpi::broadcast(world, counts.data(), num_processes, 0);

  if (world.rank() == 0) {
    input_ = std::vector<std::vector<int>>(taskData->inputs.size());
    for (size_t i = 0; i < input_.size(); i++) {
      auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
      input_[i] = std::vector<int>(taskData->inputs_count[i]);
      for (size_t j = 0; j < taskData->inputs_count[i]; j++) {
        input_[i][j] = tmp_ptr[j];
      }
    }

    size_t offset_remainder = counts[0];
    for (unsigned int proc = 1; proc < num_processes; proc++) {
      size_t current_count = counts[proc];
      world.send(proc, 0, input_[0].data() + offset_remainder, current_count);
      world.send(proc, 1, input_[1].data() + offset_remainder, current_count);
      offset_remainder += current_count;
    }
  }

  local_input1_ = std::vector<int>(counts[world.rank()]);
  local_input2_ = std::vector<int>(counts[world.rank()]);

  if (world.rank() > 0) {
    world.recv(0, 0, local_input1_.data(), counts[world.rank()]);
    world.recv(0, 1, local_input2_.data(), counts[world.rank()]);
  } else {
    local_input1_ = std::vector<int>(input_[0].begin(), input_[0].begin() + counts[0]);
    local_input2_ = std::vector<int>(input_[1].begin(), input_[1].begin() + counts[0]);
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
  boost::mpi::reduce(world, local_res, res, std::plus<>(), 0);
  return true;
}

bool rezantseva_a_vector_dot_product_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
