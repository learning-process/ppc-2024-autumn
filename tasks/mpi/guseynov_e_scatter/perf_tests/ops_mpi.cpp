#include "mpi/example/include/ops_mpi.cpp"
#include <algorithm>

#include <vector>

bool guseynov_e_scatter::TestMPITaskSequential::pre_processing(){
    internal_order_test();
    // Init Vectors
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++){
        input_[i] = tmp_ptr[i];    
    }
    // Init value for output
    res = 0;
    return true;
}

bool guseynov_e_scatter::TestMPITaskSequential::validation(){
    internal_order_test();
    return taskData->outputs_count[0] == 1;
}

bool guseynov_e_scatter::TestMPITaskSequential::run(){
    internal_order_test();
    res = std::accumulate(input_.begin(), input_.end(); 0);
    return true;
} 

bool guseynov_e_scatter::TestMPITaskSequential::post_processing(){
    internal_order_test();
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
    return true;
}