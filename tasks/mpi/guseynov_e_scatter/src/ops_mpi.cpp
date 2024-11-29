#include "mpi/guseynov_e_scatter/include/ops_mpi.hpp"
#include <algorithm>

#include <vector>

bool guseynov_e_scatter_mpi::TestMPITaskSequential::pre_processing(){
    internal_order_test();
    // Init Vectors
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++){
        input_[i] = tmp_ptr[i];    
    }
    // Init value for output
    res_ = 0;
    return true;
}

bool guseynov_e_scatter_mpi::TestMPITaskSequential::validation(){
    internal_order_test();
    return taskData->outputs_count[0] == 1;
}

bool guseynov_e_scatter_mpi::TestMPITaskSequential::run(){
    internal_order_test();
    res_ = std::accumulate(input_.begin(), input_.end(), 0);
    return true;
} 

bool guseynov_e_scatter_mpi::TestMPITaskSequential::post_processing(){
    internal_order_test();
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
    return true;
}

bool guseynov_e_scatter_mpi::TestMPITaskParallel::pre_processing(){
    internal_order_test();
    res_ = 0;
    return true;
}

bool guseynov_e_scatter_mpi::TestMPITaskParallel::validation(){
    internal_order_test();
    return taskData->outputs_count[0] == 1;
}

bool guseynov_e_scatter_mpi::TestMPITaskParallel::run(){
    internal_order_test();
    if (world.rank() == 0){
        input_ = std::vector<int>(taskData->inputs_count[0]);
        auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
        for (unsigned i = 0; i < taskData->inputs_count[0]; i++){
         input_[i] = tmp_ptr[i];    
        }
    
    }

    // Sizes for scatterv
    int local_input_size = input_.size()/world.size();
    local_input_ = std::vector<int>(local_input_size);

    boost::mpi::scatter(world, input_, local_input_.data(), local_input_size, 0);
    int local_res = std::accumulate(local_input_.begin(), local_input_.end(), 0);
    reduce(world, local_res, res_, std::plus(), 0);

    return true;
}

bool guseynov_e_scatter_mpi::TestMPITaskParallel::post_processing(){
    internal_order_test();
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
    return true;
}