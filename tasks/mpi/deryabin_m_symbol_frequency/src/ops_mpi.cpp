#include "mpi/deryabin_m_symbol_frequency/include/ops_mpi.hpp"

#include <thread>

bool deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_str_ = std::vector<char>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_str_[i] = tmp_ptr[i];
  }
  frequency_ = 0;
  input_symbol_ = reinterpret_cast<char*>(taskData->inputs[1])[0];
  return true;
}

bool deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements
  return taskData->outputs_count[0] == 1 && taskData->inputs_count[1] == 1;
}

bool deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskSequential::run() {
  internal_order_test();
  for (char i : input_str_) {
    if (i == input_symbol_) {
      frequency_++;
    }
  }
  return true;
}

bool deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = frequency_;
  return true;
}

bool deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskParallel::pre_processing() {
  internal_order_test();
  return true;
}

bool deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1 && taskData->inputs_count[1] == 1;
  }
  return true;
}

bool deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskParallel::run() {
  internal_order_test();
  unsigned int delta = 0;
  unsigned int ostatock = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
    ostatock = taskData->inputs_count[0] % world.size();
    // Init value for input
    input_symbol_ = reinterpret_cast<char*>(taskData->inputs[1])[0];
    input_str_ = std::vector<char>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_str_[i] = tmp_ptr[i];
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_str_.data() + (proc - 1) * delta, delta);
    }
  }
  boost::mpi::broadcast(world, delta, 0);
  boost::mpi::broadcast(world, input_symbol_, 0);
  local_input_str_ = std::vector<char>(delta);
  if (world.rank() == 0) {
    local_input_str_ = std::vector<char>(input_str_.end() - delta - ostatock, input_str_.end());
  } else {
    world.recv(0, 0, local_input_str_.data(), delta);
  }
  // Init value for output
  frequency_ = 0;
  // Init local value
  local_found_ = 0;
  for (char i : local_input_str_) {
    if (i == input_symbol_) {
      local_found_++;
    }
  }
  boost::mpi::reduce(world, local_found_, frequency_, std::plus<>(), 0);
  return true;
}

bool deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = frequency_;
  }
  return true;
}
