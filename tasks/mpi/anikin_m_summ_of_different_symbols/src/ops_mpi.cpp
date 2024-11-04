// Copyright 2024 Anikin Maksim

#include "mpi/anikin_m_summ_of_different_symbols/include/ops_mpi.hpp"

#include <boost/mpi.hpp>
#include <string>
bool anikin_m_summ_of_different_symbols_mpi::SumDifSymMPISequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == taskData->inputs_count[1] && taskData->outputs_count[0] == 1;
}

bool anikin_m_summ_of_different_symbols_mpi::SumDifSymMPISequential::pre_processing() {
  internal_order_test();
  input_a.assign(reinterpret_cast<char *>(taskData->inputs[0]), taskData->inputs_count[0]);
  input_b.assign(reinterpret_cast<char *>(taskData->inputs[1]), taskData->inputs_count[1]);
  result_ = 0;
  return true;
}

bool anikin_m_summ_of_different_symbols_mpi::SumDifSymMPISequential::run() {
  internal_order_test();
  for (size_t i = 0; i < input_a.size(); ++i) {
    if (input_a[i] != input_b[i]) {
      result_++;
    }
  }
  return true;
}

bool anikin_m_summ_of_different_symbols_mpi::SumDifSymMPISequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<int *>(taskData->outputs[0]) = result_;
  return true;
}

bool anikin_m_summ_of_different_symbols_mpi::SumDifSymMPIParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] == taskData->inputs_count[1] && taskData->outputs_count[0] == 1;
  }
  return true;
}

bool anikin_m_summ_of_different_symbols_mpi::SumDifSymMPIParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_a.assign(reinterpret_cast<char *>(taskData->inputs[0]), taskData->inputs_count[0]);
    input_b.assign(reinterpret_cast<char *>(taskData->inputs[1]), taskData->inputs_count[1]);
    auto base_size = input_a.size() / world.size();
    auto remainder = input_a.size() % world.size();
    local_input_a = input_a.substr(0, base_size);
    local_input_b = input_b.substr(0, base_size);

    auto start = base_size;
    for (int p = 1; p < world.size(); ++p) {
      auto size = base_size + (p <= int(remainder) ? 1 : 0);

      world.send(p, 0, input_a.substr(start, size));
      world.send(p, 0, input_b.substr(start, size));

      start += size;
    }
  } else {
    world.recv(0, 0, local_input_a);
    world.recv(0, 0, local_input_b);
  }

  result_ = 0;
  return true;
}

bool anikin_m_summ_of_different_symbols_mpi::SumDifSymMPIParallel::run() {
  internal_order_test();
  auto local_result = 0;
  for (size_t i = 0; i < local_input_a.size(); ++i) {
    if (local_input_a[i] != local_input_b[i]) {
      local_result++;
    }
  }
  boost::mpi::reduce(world, local_result, result_, std::plus(), 0);
  return true;
}

bool anikin_m_summ_of_different_symbols_mpi::SumDifSymMPIParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<int *>(taskData->outputs[0]) = result_;
  }
  return true;
}