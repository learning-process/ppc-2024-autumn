#include "seq/deryabin_m_symbol_frequency/include/ops_seq.hpp"

#include <thread>

bool deryabin_m_symbol_frequency_seq::SymbolFrequencyTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_str_ = reinterpret_cast<std::string*>(taskData->inputs[0])[0];
  frequency_ = 0;
  input_symbol_ = reinterpret_cast<char*>(taskData->inputs[1])[0];
  return true;
}

bool deryabin_m_symbol_frequency_seq::SymbolFrequencyTaskSequential::validation() {
  internal_order_test();
  // Check count elements
  if (taskData->inputs_count[0] != 1 || taskData->outputs_count[0] != 1 || taskData->inputs_count[1] != 1) {
    return false;
  };
  return true;
}

bool deryabin_m_symbol_frequency_seq::SymbolFrequencyTaskSequential::run() {
  internal_order_test();
  for (char i : input_str_) {
    if (i == input_symbol_) {
      frequency_++;
    }
  }
  return true;
}

bool deryabin_m_symbol_frequency_seq::SymbolFrequencyTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = frequency_;
  return true;
}
