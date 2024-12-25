// Copyright 2023 Nesterov Alexander
#include "mpi/gorbunov_e_check_lexicographic/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;
/*
SEQ
*/
bool gorbunov_e_check_lexicographic_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  
  input_left_ = std::vector<char>(taskData->inputs_count[0]);
  input_right_ = std::vector<char>(taskData->inputs_count[1]);

  auto* tmp_ptr_left = reinterpret_cast<char*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_left_[i] = tmp_ptr_left[i];
  }
  
  auto* tmp_ptr_right = reinterpret_cast<char*>(taskData->inputs[1]);
  for (unsigned i = 0; i < taskData->inputs_count[1]; i++) {
    input_right_[i] = tmp_ptr_right[i];
  }

  // Init value for output
  res = 0;
  return true;
}

bool gorbunov_e_check_lexicographic_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1;
}

bool gorbunov_e_check_lexicographic_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  int result = -1;
  int iterLim = std::min(input_left_.size(), input_right_.size());
  int iterFor = std::max(input_left_.size(), input_right_.size());

  for (size_t i = 0; i < iterFor - 1 && result = -1; i++)
  {
    // case with different len of strs
    if (i >= iterLim) {
      result = i;
    } else 
    // case with difference in chars
    if (static_cast<int>(input_[i]) > static_cast<int>(input_[i+1]))
      result = i;
  }
  
  res = result;
  return true;
}

bool gorbunov_e_check_lexicographic_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

/*
MPI
*/
bool gorbunov_e_check_lexicographic_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  unsigned int step = 0; // length of data for non-0-processes (e.g. step=10 means 1, 2, 3, etc. processes will work on 10 symbols)
  unsigned int delta = 0; // diff bitwin 0-process data and non-0-process data (e.g. delta=3 means 0 process will work on 13 symbols when step=10)

  unsigned int iterFor;

  if (world.rank() == 0) {
    iterFor = std::max(taskData->inputs_count[0], taskData->inputs_count[1]);

    step = iterFor / world.size();
    delta = iterFor % world.size();
  }

  // comm, data, root
  broadcast(world, step, 0);

  if (world.rank() == 0) {
    
    input_left_ = std::vector<char>(iterFor);
    input_right_ = std::vector<char>(iterFor);

    auto* tmp_ptr_left = reinterpret_cast<char*>(taskData->inputs[0]);
    auto* tmp_ptr_right = reinterpret_cast<char*>(taskData->inputs[1]);

    for (unsigned i = 0; i < iterFor; i++) {
      if (i >= taskData->inputs_count[0]) input_left_[i] = '\0';
      else input_left_[i] = tmp_ptr_left[i];

      if (i >= taskData->inputs_count[1]) input_right_[i] = '\0';
      else input_right_[i] = tmp_ptr_right[i];
    }
  
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_left_.data() + proc * step + delta, step);
      world.send(proc, 1, input_right_.data() + proc * step + delta, step);
    }
  }

  local_input_left = std::vector<char>(step);
  local_input_right = std::vector<char>(step);
  if (world.rank() == 0) {
    local_input_left = std::vector<char>(input_left_.begin(), input_left_.begin() + delta + step);
    local_input_right = std::vector<char>(input_right_.begin(), input_right_.begin() + delta + step);
  } else {
    // from, tag, data, size
    world.recv(0, 0, local_input_left.data(), step);
    world.recv(0, 1, local_input_right.data(), step);
  }

  res = true;
  return true;
}

bool gorbunov_e_check_lexicographic_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool gorbunov_e_check_lexicographic_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  
  // computing local results
  std::vector<int> local_res = std::vector<int>(1);

  int result = -1;
  for (size_t i = 0; i < local_input_left.size() - 1 && result < 0; i++)
  {
    if (local_input_left[i] != local_input_right[i])
      result = i;
  }

  // actually local for non-0-processes; "previous" for 0-process
  local_res[0] = static_cast<int>(result);

  // sending local results for non-0-processes
  if (world.rank() > 0) {
    world.send(0, 77, local_res, 1);
  } 
  // receiving local result for 0-process & computing global result
  else {
    if (local_res[0] >= 0) {
      res = local_res[0];
      return true;
    }

    for (size_t i = 1; i < world.size() && result < 0; i++)
    {
      world.recv(i, 77, local_res, 1);

      unsigned int iterFor = std::max(taskData->inputs_count[0], taskData->inputs_count[1]);
      step = iterFor / world.size();
      delta = iterFor % world.size();

      if (local_res[0] >= 0) {
        res = delta + step * i + local_res[0];
        return true;
      }
    }

    res = 0;
  }

  return true;
}

bool gorbunov_e_check_lexicographic_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<bool*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
