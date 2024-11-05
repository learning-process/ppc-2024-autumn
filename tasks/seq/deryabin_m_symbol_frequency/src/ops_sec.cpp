#include "seq/deryabin_m_symbol_frequency/include/ops_sec.hpp"

#include <thread>

using namespace std::chrono_literals;

#include <string>

bool deryabin_m_symbol_frequency_sec::SymbolFrequencyTaskSequential::pre_processing() {
	internal_order_test();
	// Init value for input and output
	input_str_ = reinterpret_cast<std::string*>(taskData->inputs[0])[0];
	frequency_ = 0;
	input_symbol_ = reinterpret_cast<char*>(taskData->inputs[1])[0];
	return true;
}

bool deryabin_m_symbol_frequency_sec::SymbolFrequencyTaskSequential::validation() {
	internal_order_test();
	// Check count elements of output
	return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1 && taskData->inputs_count[1] == 1;
}

bool deryabin_m_symbol_frequency_sec::SymbolFrequencyTaskSequential::run() {
	internal_order_test();
	double found = 0;
	for (char i : input_str_)
		if (i == input_symbol_) found++;
	if (input_str_.size() != 0)
		frequency_ = found / input_str_.size();
	return true;
}

bool deryabin_m_symbol_frequency_sec::SymbolFrequencyTaskSequential::post_processing() {
	internal_order_test();
	reinterpret_cast<double*>(taskData->outputs[0])[0] = frequency_;
	return true;
}
