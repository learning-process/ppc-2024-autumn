#include "mpi/kudryashova_i_vector_dot_product/include/vectorDotProductMPI.hpp"

#include <boost/mpi.hpp>

int kudryashova_i_vector_dot_product_mpi::vectorDotProduct(const std::vector<int>& vector1,
                                                           const std::vector<int>& vector2) {
  long long result = 0;
  for (unsigned long i = 0; i < vector1.size(); i++) result += vector1[i] * vector2[i];
  return result;
}

bool kudryashova_i_vector_dot_product_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  input_.resize(taskData->inputs.size());
  for (unsigned long i = 0; i < input_.size(); ++i) {
    auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[i]);
    input_[i] = std::vector<int>(taskData->inputs_count[i]);
    std::copy(tempPtr, tempPtr + taskData->inputs_count[i], input_[i].begin());
  }
  return true;
}

bool kudryashova_i_vector_dot_product_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] == taskData->inputs_count[1]) &&
         (taskData->inputs.size() == taskData->inputs_count.size() && taskData->inputs.size() == 2) &&
         taskData->outputs_count[0] == 1 && (taskData->outputs.size() == taskData->outputs_count.size()) &&
         taskData->outputs.size() == 1 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0;
}

bool kudryashova_i_vector_dot_product_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (unsigned long i = 0; i < input_[0].size(); i++) {
    result += input_[1][i] * input_[0][i];
  }
  return true;
}

bool kudryashova_i_vector_dot_product_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
  return true;
}

bool kudryashova_i_vector_dot_product_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
    if ((int)(taskData->inputs_count[0]) < world.size()) {
      delta = taskData->inputs_count[0];
    }
  }
  if (world.rank() == 0) {
    input_.resize(taskData->inputs.size());
    for (size_t i = 0; i < taskData->inputs.size(); ++i) {
      if (taskData->inputs[i] == nullptr || taskData->inputs_count[i] == 0) {
        return false;
      }
      input_[i].resize(taskData->inputs_count[i]);
      int* source_ptr = reinterpret_cast<int*>(taskData->inputs[i]);

      std::copy(source_ptr, source_ptr + taskData->inputs_count[i], input_[i].begin());
    }
  }
  return true;
}

bool kudryashova_i_vector_dot_product_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (taskData->inputs_count[0] == taskData->inputs_count[1]) &&
           (taskData->inputs.size() == taskData->inputs_count.size() && taskData->inputs.size() == 2) &&
           taskData->outputs_count[0] == 1 && (taskData->outputs.size() == taskData->outputs_count.size()) &&
           taskData->outputs.size() == 1 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0;
  }
  return true;
}

bool kudryashova_i_vector_dot_product_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  broadcast(world, delta, 0);
  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); ++proc) {
      world.send(proc, 0, input_[0].data() + proc * delta, delta);
      world.send(proc, 1, input_[1].data() + proc * delta, delta);
    }
  }
  local_input1_.resize(delta);
  local_input2_.resize(delta);
  if (world.rank() == 0) {
    std::copy(input_[0].begin(), input_[0].begin() + delta, local_input1_.begin());
    std::copy(input_[1].begin(), input_[1].begin() + delta, local_input2_.begin());
  } else {
    world.recv(0, 0, local_input1_.data(), delta);
    world.recv(0, 1, local_input2_.data(), delta);
  }
  int local_result = std::inner_product(local_input1_.begin(), local_input1_.end(), local_input2_.begin(), 0);
  std::vector<int> full_results;
  gather(world, local_result, full_results, 0);

  if (world.rank() == 0) {
    result = std::accumulate(full_results.begin(), full_results.end(), 0);
  }
  if (world.rank() == 0 && (int)(taskData->inputs_count[0]) < world.size()) {
    result = std::inner_product(input_[0].begin(), input_[0].end(), input_[1].begin(), 0);
  }
  return true;
}

bool kudryashova_i_vector_dot_product_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    if (!taskData->outputs.empty()) {
      reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
    } else {
      return false;
    }
  }
  return true;
}