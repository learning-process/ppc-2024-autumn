#include "mpi/deryabin_m_symbol_frequency/include/ops_mpi.hpp"

#include <functional>
#include <string>
#include <thread>

using namespace std::chrono_literals;

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
    float found = 0;
    for (char i : input_str_)
        if (i == input_symbol_) found++;
    if (taskData->inputs_count[0] != 0)
        frequency_ = found / taskData->inputs_count[0];
    return true;
}

bool deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskSequential::post_processing() {
    internal_order_test();
    reinterpret_cast<float*>(taskData->outputs[0])[0] = frequency_;
    return true;
}

bool deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskParallel::pre_processing() {
    internal_order_test();
    unsigned int delta = 0;
    unsigned int ostatock = 0;
    if (world.rank() == 0) {
        delta = taskData->inputs_count[0] / world.size();
        ostatock = taskData->inputs_count[0] % world.size();
        // Init value for input
        input_symbol_ = reinterpret_cast<char*>(taskData->inputs[1])[0];
    }
    boost::mpi::broadcast(world, delta, 0);
    boost::mpi::broadcast(world, ostatock, 0);
    boost::mpi::broadcast(world, input_symbol_, 0);
    local_input_str_ = std::vector<char>(delta);
    if (world.rank() == 0) {
        // Init value for input
        input_str_ = std::vector<char>(taskData->inputs_count[0]);
        auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
        for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
            input_str_[i] = tmp_ptr[i];
        }
        for (int proc = 1; proc < world.size(); proc++) {
            if (proc == world.size() - 1)
            {
                local_input_str_.resize(delta + ostatock);
                world.send(proc, 0, input_str_.data() + proc * delta, delta + ostatock);
            }
            else
            {
                world.send(proc, 0, input_str_.data() + proc * delta, delta);
            }
        }
    }
    if (world.rank() == world.size() - 1) {
        world.recv(0, 0, local_input_str_.data(), delta + ostatock);
    }
    else {
        world.recv(0, 0, local_input_str_.data(), delta);
    }
    local_found_ = 0;
    found_ = 0;
    // Init value for output
    frequency_ = 0;
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
    for (char i : local_input_str_)
        if (i == input_symbol_) local_found_++;
    boost::mpi::reduce(world, local_found_, found_, std::plus<>(), 0);
    if (taskData->inputs_count[0] != 0)
        frequency_ = float(found_) / taskData->inputs_count[0];
    return true;
}

bool deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskParallel::post_processing() {
    internal_order_test();
    if (world.rank() == 0) {
        reinterpret_cast<float*>(taskData->outputs[0])[0] = frequency_;
    }
    return true;
}
