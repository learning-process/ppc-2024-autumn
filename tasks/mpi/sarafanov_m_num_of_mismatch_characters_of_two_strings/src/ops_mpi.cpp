#include "mpi/sarafanov_m_num_of_mismatch_characters_of_two_strings/include/ops_mpi.hpp"

#include <boost/mpi.hpp>
#include <string>

bool sarafanov_m_num_of_mismatch_characters_of_two_strings_mpi::SequentialTask::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == taskData->inputs_count[1] && taskData->outputs_count[0] == 1;
}

bool sarafanov_m_num_of_mismatch_characters_of_two_strings_mpi::SequentialTask::pre_processing() {
  internal_order_test();
  input_a_.assign(reinterpret_cast<char *>(taskData->inputs[0]), taskData->inputs_count[0]);
  input_b_.assign(reinterpret_cast<char *>(taskData->inputs[1]), taskData->inputs_count[1]);
  result_ = 0;
  return true;
}

bool sarafanov_m_num_of_mismatch_characters_of_two_strings_mpi::SequentialTask::run() {
  internal_order_test();
  for (size_t i = 0; i < input_a_.size(); ++i) {
    if (input_a_[i] != input_b_[i]) {
      result_++;
    }
  }
  return true;
}

bool sarafanov_m_num_of_mismatch_characters_of_two_strings_mpi::SequentialTask::post_processing() {
  internal_order_test();
  *reinterpret_cast<int *>(taskData->outputs[0]) = result_;
  return true;
}

bool sarafanov_m_num_of_mismatch_characters_of_two_strings_mpi::ParallelTask::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] == taskData->inputs_count[1] && taskData->outputs_count[0] == 1;
  }
  return true;
}

bool sarafanov_m_num_of_mismatch_characters_of_two_strings_mpi::ParallelTask::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_a_.assign(reinterpret_cast<char *>(taskData->inputs[0]), taskData->inputs_count[0]);
    input_b_.assign(reinterpret_cast<char *>(taskData->inputs[1]), taskData->inputs_count[1]);
    result_ = 0;
  }
  return true;
}

bool sarafanov_m_num_of_mismatch_characters_of_two_strings_mpi::ParallelTask::run() {
  internal_order_test();
  std::string local_input_a;
  std::string local_input_b;
  if (world.rank() == 0) {
    auto base_size = input_a_.size() / world.size();
    auto remainder = input_a_.size() % world.size();
    local_input_a = input_a_.substr(0, base_size);
    local_input_b = input_b_.substr(0, base_size);

    auto start = base_size;
    for (int p = 1; p < world.size(); ++p) {
      auto size = base_size + (p <= int(remainder) ? 1 : 0);

      world.send(p, 0, input_a_.substr(start, size));
      world.send(p, 0, input_b_.substr(start, size));

      start += size;
    }
  } else {
    world.recv(0, 0, local_input_a);
    world.recv(0, 0, local_input_b);
  }

  auto local_result = 0;
  for (size_t i = 0; i < local_input_a.size(); ++i) {
    if (local_input_a[i] != local_input_b[i]) {
      local_result++;
    }
  }
  boost::mpi::reduce(world, local_result, result_, std::plus(), 0);
  return true;
}

bool sarafanov_m_num_of_mismatch_characters_of_two_strings_mpi::ParallelTask::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<int *>(taskData->outputs[0]) = result_;
  }
  return true;
}
