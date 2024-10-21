#pragma once

#include <gtest/gtest.h>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace sotskov_a_sum_element_matrix_mpi {

template <typename T>
std::vector<T> create_random_matrix(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        return {};
    }

    std::vector<T> matrix(rows * cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-100, 100);

    std::generate(matrix.begin(), matrix.end(), [&]() { return static_cast<T>(dis(gen)); });
    return matrix;
}

template <typename T>
T sum_matrix_elements(const std::vector<T>& matrix) {
    return std::accumulate(matrix.begin(), matrix.end(), T(0));
}

template <typename T>
class TestMPITaskSequential : public ppc::core::Task {
 public:
    explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> task_data) 
        : Task(std::move(task_data)) {}

    bool pre_processing() override; 
    bool validation() override;      
    bool run() override;             
    bool post_processing() override;  

 private:
    std::vector<T> input_data_; 
    T result_{0};  
};

template <typename T>
bool TestMPITaskSequential<T>::pre_processing() {
    internal_order_test();
    result_ = 0;
    T* tmp_ptr = reinterpret_cast<T*>(taskData->inputs[0]);
    input_data_.assign(tmp_ptr, tmp_ptr + taskData->inputs_count[0]);
    return true;
}

template <typename T>
bool TestMPITaskSequential<T>::validation() {
    internal_order_test();
    return taskData->outputs_count[0] == 1;
}

template <typename T>
bool TestMPITaskSequential<T>::run() {
    internal_order_test();
    result_ = std::accumulate(input_data_.begin(), input_data_.end(), T(0));
    return true;
}

template <typename T>
bool TestMPITaskSequential<T>::post_processing() {
    internal_order_test();
    if (!taskData->outputs.empty() && taskData->outputs[0] != nullptr) {
        reinterpret_cast<T*>(taskData->outputs[0])[0] = result_;
        return true;
    }
    return false;
}

template <typename T>
class TestMPITaskParallel : public ppc::core::Task {
 public:
    explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
        : Task(std::move(taskData_)), world(boost::mpi::communicator()) {}

    bool pre_processing() override; 
    bool validation() override;      
    bool run() override;            
    bool post_processing() override; 

 private:
    std::vector<T> input_data, local_input_;  
    T result_{};  
    int rows{};   
    int columns{}; 
    int delta{};
    boost::mpi::communicator world; 
};

template <typename T>
bool TestMPITaskParallel<T>::pre_processing() {
    internal_order_test();

    delta = 0;
    if (world.rank() == 0) {
      result_ = 0;
      rows = static_cast<int>(taskData->inputs_count[1]);
      columns = static_cast<int>(taskData->inputs_count[2]);
      delta = (rows * columns) / world.size();
    }

    boost::mpi::broadcast(world, delta, 0);

    if (world.rank() == 0) {
      input_data = std::vector<T>(rows * columns);
      auto* tmp_ptr = reinterpret_cast<T*>(taskData->inputs[0]);
      for (int i = 0; i < static_cast<int>(taskData->inputs_count[0]); i++) {
        input_data[i] = tmp_ptr[i];
      }
      for (int proc = 1; proc < world.size(); proc++) {
        world.send(proc, 0, input_data.data() + proc * delta, delta);
      }
    }

    local_input_ = std::vector<T>(delta);
    if (world.rank() == 0) {
      local_input_ = std::vector<T>(input_data.begin(), input_data.begin() + delta);
    } else {
      world.recv(0, 0, local_input_.data(), delta);
    }
    result_ = 0;
    return true;
}


template <typename T>
bool TestMPITaskParallel<T>::validation() {
    internal_order_test();

    if (world.rank() == 0) {
      return (taskData->outputs_count[0] == 1 && !(taskData->inputs.empty()));
    }
    return true;
}

template <typename T>
bool TestMPITaskParallel<T>::run() {
    internal_order_test();

    T local_res = std::accumulate(local_input_.begin(), local_input_.end(), 0);
    reduce(world, local_res, result_, std::plus<T>(), 0);

    return true;
}

template <typename T>
bool TestMPITaskParallel<T>::post_processing() {
    internal_order_test();
    if (world.rank() == 0) {
      reinterpret_cast<T*>(taskData->outputs[0])[0] = result_;
    }
    return true;
}

template class sotskov_a_sum_element_matrix_mpi::TestMPITaskSequential<int>;
template class sotskov_a_sum_element_matrix_mpi::TestMPITaskSequential<double>;
template class sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel<int>;
template class sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel<double>;

}  // namespace sotskov_a_sum_element_matrix
