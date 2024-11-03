#include "mpi/nasedkin_e_matrix_column_max_value/include/ops_mpi.hpp"
#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

std::vector<int> nasedkin_e_matrix_column_max_value_mpi::getRandomVector(int sz) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::vector<int> vec(sz);
    for (int i = 0; i < sz; i++) {
        vec[i] = gen() % 100;
    }
    return vec;
}

bool nasedkin_e_matrix_column_max_value_mpi::MatrixColumnMaxTaskSequential::pre_processing() {
    input_.resize(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
    res = 0;
    return true;
}

bool nasedkin_e_matrix_column_max_value_mpi::MatrixColumnMaxTaskSequential::validation() {
    return taskData->outputs_count[0] == 1;
}

bool nasedkin_e_matrix_column_max_value_mpi::MatrixColumnMaxTaskSequential::run() {
    res = *std::max_element(input_.begin(), input_.end());
    return true;
}

bool nasedkin_e_matrix_column_max_value_mpi::MatrixColumnMaxTaskSequential::post_processing() {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
    return true;
}

bool nasedkin_e_matrix_column_max_value_mpi::MatrixColumnMaxTaskParallel::pre_processing() {
    unsigned int delta = 0;
    if (world.rank() == 0) {
        delta = taskData->inputs_count[0] / world.size();
    }
    boost::mpi::broadcast(world, delta, 0);

    if (world.rank() == 0) {
        input_.resize(taskData->inputs_count[0]);
        auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
        std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
        for (int proc = 1; proc < world.size(); proc++) {
            world.send(proc, 0, input_.data() + proc * delta, delta);
        }
    }
    local_input_.resize(delta);
    if (world.rank() == 0) {
        local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta);
    } else {
        world.recv(0, 0, local_input_.data(), delta);
    }
    res = 0;
    return true;
}

bool nasedkin_e_matrix_column_max_value_mpi::MatrixColumnMaxTaskParallel::validation() {
    if (world.rank() == 0) {
        return taskData->outputs_count[0] == 1;
    }
    return true;
}

bool nasedkin_e_matrix_column_max_value_mpi::MatrixColumnMaxTaskParallel::run() {
    int local_res = *std::max_element(local_input_.begin(), local_input_.end());
    boost::mpi::reduce(world, local_res, res, boost::mpi::maximum<int>(), 0);
    return true;
}

bool nasedkin_e_matrix_column_max_value_mpi::MatrixColumnMaxTaskParallel::post_processing() {
    if (world.rank() == 0) {
        reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
    }
    return true;
}
