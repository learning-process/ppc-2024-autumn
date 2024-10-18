#include "mpi/chernykh_a_num_of_alternations_signs/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <random>
#include <vector>

std::vector<int> chernykh_a_num_of_alternations_signs_mpi::getRandomVector(size_t size) {
  auto dev = std::random_device();
  auto gen = std::mt19937(dev());
  auto dist = std::uniform_int_distribution<int>(-100'000, 100'000);
  auto result = std::vector<int>(size);
  for (auto& val : result) {
    val = dist(gen);
  }
  return result;
}

bool chernykh_a_num_of_alternations_signs_mpi::SequentialTask::validation() {
  internal_order_test();
  return taskData->inputs_count[0] >= 2 && taskData->outputs_count[0] == 1;
}

bool chernykh_a_num_of_alternations_signs_mpi::SequentialTask::pre_processing() {
  internal_order_test();
  auto* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  auto input_size = taskData->inputs_count[0];
  input = std::vector<int>(input_ptr, input_ptr + input_size);
  result = 0;
  return true;
}

bool chernykh_a_num_of_alternations_signs_mpi::SequentialTask::run() {
  internal_order_test();
  auto input_size = input.size();
  for (size_t i = 0; i < input_size - 1; i++) {
    if ((input[i] ^ input[i + 1]) < 0) {
      result++;
    }
  }
  return true;
}

bool chernykh_a_num_of_alternations_signs_mpi::SequentialTask::post_processing() {
  internal_order_test();
  *reinterpret_cast<int*>(taskData->outputs[0]) = result;
  return true;
}

bool chernykh_a_num_of_alternations_signs_mpi::ParallelTask::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] >= 2 && taskData->outputs_count[0] == 1;
  }
  return true;
}

bool chernykh_a_num_of_alternations_signs_mpi::ParallelTask::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto input_size = taskData->inputs_count[0];
    auto chunk_size = input_size / world.size();
    auto* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    input = std::vector<int>(input_ptr, input_ptr + input_size);
    chunk = std::vector<int>(input_ptr, input_ptr + chunk_size + uint32_t(world.size() > 1));

    for (int proc = 1; proc < world.size(); proc++) {
      auto start = proc * chunk_size;
      auto size = (proc == world.size() - 1) ? input_size - start : chunk_size + 1;
      world.send(proc, 0, std::vector<int>(input_ptr + start, input_ptr + start + size));
    }
  } else {
    world.recv(0, 0, chunk);
  }

  result = 0;
  return true;
}

bool chernykh_a_num_of_alternations_signs_mpi::ParallelTask::run() {
  internal_order_test();
  auto chunk_result = 0;
  auto chunk_size = chunk.size();
  for (size_t i = 0; i < chunk_size - 1; i++) {
    if ((chunk[i] ^ chunk[i + 1]) < 0) {
      chunk_result++;
    }
  }
  boost::mpi::reduce(world, chunk_result, result, std::plus(), 0);
  return true;
}

bool chernykh_a_num_of_alternations_signs_mpi::ParallelTask::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<int*>(taskData->outputs[0]) = result;
  }
  return true;
}
