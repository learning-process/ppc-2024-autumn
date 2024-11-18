#include "mpi/petrov_o_num_of_alternations_signs/include/ops_mpi.hpp"

#include <boost/mpi/datatype.hpp>
#include <vector>

bool petrov_o_num_of_alternations_signs_mpi::ParallelTask::pre_processing() {
  internal_order_test();
  this->res = 0;
  return true;
}

bool petrov_o_num_of_alternations_signs_mpi::ParallelTask::validation() {
  internal_order_test();

  if (world.rank() != 0) return true;
  return taskData->outputs_count[0] == 1;
}

bool petrov_o_num_of_alternations_signs_mpi::ParallelTask::run() {
  internal_order_test();

  int input_size = 0;

  if (world.rank() == 0) {
    input_size = taskData->inputs_count[0];
  }

  int active_processes = std::min((int)world.size(), input_size);  // Number of active processes

  boost::mpi::broadcast(world, input_size, 0);
  boost::mpi::broadcast(world, active_processes, 0);

  if (input_size < 2) {
    this->res = 0;
    return true;
  }

  if (world.rank() >= active_processes) {  // end work for all unused processes
    return true;
  }

  if (world.rank() == 0) {
    const int* input = reinterpret_cast<int*>(taskData->inputs[0]);
    this->input_.resize(input_size);
    std::copy(input, input + input_size, std::begin(this->input_));

    std::vector<int> distribution(active_processes);
    std::vector<int> displacement(active_processes);

    int chunk_size = input_size / active_processes;
    int remainder = input_size % active_processes;

    for (int i = 0; i < active_processes; ++i) {
      distribution[i] = chunk_size + static_cast<int>(i < remainder);  // Distribute remainder
      displacement[i] = (i == 0) ? 0 : displacement[i - 1] + distribution[i - 1];
    }

    chunk.resize(distribution[world.rank()]);

    boost::mpi::scatterv(world, input, distribution, displacement, chunk.data(), distribution[world.rank()], 0);

  } else {
    int chunk_size = input_size / active_processes;
    int remainder = input_size % active_processes;

    int distribution = chunk_size + static_cast<int>(world.rank() < remainder);

    chunk.resize(distribution);

    int input;  // clang-tidy needs unused input
    boost::mpi::scatterv(world, &input, chunk.data(), distribution, 0);
  }

  auto local_res = 0;

  for (size_t i = 1; i < chunk.size(); i++) {
    if ((chunk[i] < 0) ^ (chunk[i - 1] < 0)) {
      local_res++;
    }
  }

  int last_element = chunk.back();
  int next_element = 0;

  if (world.rank() < active_processes - 1) {
    world.send(world.rank() + 1, 0, last_element);
  }

  if (world.rank() > 0) {
    world.recv(world.rank() - 1, 0, next_element);
    if ((chunk.front() < 0) ^ (next_element < 0)) {
      local_res++;
    }
  }

  boost::mpi::reduce(world, local_res, res, std::plus(), 0);
  return true;
}

bool petrov_o_num_of_alternations_signs_mpi::ParallelTask::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<int*>(taskData->outputs[0]) = res;
  }
  return true;
}

bool petrov_o_num_of_alternations_signs_mpi::SequentialTask::pre_processing() {
  internal_order_test();

  const auto input_size = taskData->inputs_count[0];

  const int* input = reinterpret_cast<int*>(taskData->inputs[0]);
  this->input_.resize(input_size);
  std::copy(input, input + input_size, std::begin(this->input_));

  this->res = 0;

  return true;
}

bool petrov_o_num_of_alternations_signs_mpi::SequentialTask::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool petrov_o_num_of_alternations_signs_mpi::SequentialTask::run() {
  internal_order_test();

  if (input_.size() > 1) {
    for (size_t i = 1; i < input_.size(); i++) {
      if ((input_[i] < 0) ^ (input_[i - 1] < 0)) {
        this->res++;
      }
    }
  }

  return true;
}

bool petrov_o_num_of_alternations_signs_mpi::SequentialTask::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
