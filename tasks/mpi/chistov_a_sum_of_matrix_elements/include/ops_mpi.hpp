#pragma once
#include <gtest/gtest.h>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"


namespace chistov_a_sum_of_matrix_elements {

template <typename T = int>
void print_matrix(const std::vector<T> matrix, const int n, const int m) {
  std::cout << "Matrix:" << std::endl;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) std::cout << matrix[i * m + j] << " ";
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template <typename T = int>
std::vector<T> getRandomMatrix(const int n, const int m) {
  if (n <= 0 || m <= 0) {
    throw std::invalid_argument("Incorrect entered N or M");
  }

  std::vector<T> matrix(n * m);

  for (int i = 0; i < n * m; ++i) {
    matrix[i] = static_cast<T>((std::rand() % 201) - 100);
  }

  return matrix;
}


template <typename T = int>
T classic_way(const std::vector<T> matrix, const int n, const int m) {
  T result = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      result += matrix[i * m + j];
    }
  }

  return result;
}

template <typename T=int>
class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_, const int n_, const int m_)
      : Task(std::move(taskData_)), n(n_), m(m_) {}

  bool pre_processing() override {
    internal_order_test();

    T* tmp_ptr = reinterpret_cast<T*>(taskData->inputs[0]);
    input_.resize(taskData->inputs_count[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());

    return true;
  }

  bool validation() override {
    internal_order_test();

    return taskData->outputs_count[0] == 1;
  }


  bool run() override {
  internal_order_test();

  res = std::accumulate(input_.begin(), input_.end(), 0);
  return true;
}

  bool post_processing() override {
  internal_order_test();

  if (taskData->outputs.size() > 0 && taskData->outputs[0] != nullptr) {
    reinterpret_cast<T*>(taskData->outputs[0])[0] = res;
    return true;
  } else {
    return false;
  }
}


 private:
  std::vector<T> input_;
  int n, m;
  T res = 0;
};

template <typename T = int>
class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, const int n_, const int m_) : Task(std::move(taskData_)), n(n_), m(m_), world(boost::mpi::communicator()) {}

bool pre_processing() override {
    internal_order_test();

    size_t delta = 0;  
    if (world.rank() == 0) {
      delta = (n * m) / world.size(); 
    }
    boost::mpi::broadcast(world, delta, 0);

    if (world.rank() == 0) {
      input_ = std::vector<T>(n * m);
      auto* tmp_ptr = reinterpret_cast<T*>(taskData->inputs[0]);
      for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
        input_[i] = tmp_ptr[i];
      }
      for (size_t proc = 1; proc < world.size(); proc++) {
        world.send(proc, 0, input_.data() + proc * delta, delta);
      }
    }

    local_input_ = std::vector<T>(delta);
    if (world.rank() == 0) {
      local_input_ = std::vector<T>(input_.begin(), input_.begin() + delta);
    } else {
      world.recv(0, 0, local_input_.data(), delta);
    }

    res = 0;
    return true;
  }

  bool validation() override {
    internal_order_test();

    if (world.rank() == 0) {
      return taskData->outputs_count[0] == 1;
    }
    return true;
  }


  bool run() override {
  internal_order_test();

  T local_res = std::accumulate(local_input_.begin(), local_input_.end(), 0);
  reduce(world, local_res, res, std::plus<T>(), 0);

  return true;
}

  bool post_processing() override {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<T*>(taskData->outputs[0])[0] = res;
  }
  return true;
}

 private:
  std::vector<T> input_, local_input_;
  T res = 0;
  int n, m;
  boost::mpi::communicator world;
};

} 




