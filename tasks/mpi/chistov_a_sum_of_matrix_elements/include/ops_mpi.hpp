// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace chistov_a_sum_of_matrix_elements {

template <typename T>
void print_matrix(const std::vector<T> matrix, const int n, const int m) {
  std::cout << "Matrix:" << std::endl;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      std::cout << matrix[i * m + j] << " ";
    }
  }
  std::cout << std::endl;
}

template <typename T>
std::vector<T> get_random_matrix(const int n, const int m) {
  if (n <= 0 || m <= 0) {
    return std::vector<T>();
  }

  std::vector<T> matrix(n * m);

  for (int i = 0; i < n * m; ++i) {
    matrix[i] = static_cast<T>((std::rand() % 201) - 100);
  }

  return matrix;
}

template <typename T>
T classic_way(const std::vector<T> matrix, const int n, const int m) {
  T result = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      result += matrix[i * m + j];
    }
  }

  return result;
}

template <typename T>
class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override {
    internal_order_test();
    res = 0;
    T* tmp_ptr = reinterpret_cast<T*>(taskData->inputs[0]);
    input_.assign(tmp_ptr, tmp_ptr + taskData->inputs_count[0]);
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

    reinterpret_cast<T*>(taskData->outputs[0])[0] = res;
    return true;
  }

 private:
  std::vector<T> input_;
  T res{};
};

template <typename T>
class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override {
    internal_order_test();

    int delta = 0;
    if (world.rank() == 0) {
      res = 0;
      n = static_cast<int>(taskData->inputs_count[1]);
      m = static_cast<int>(taskData->inputs_count[2]);
      delta = (n * m) / world.size();
    }

    boost::mpi::broadcast(world, delta, 0);

    if (world.rank() == 0) {
      input_ = std::vector<T>(n * m);
      auto* tmp_ptr = reinterpret_cast<T*>(taskData->inputs[0]);
      for (int i = 0; i < static_cast<int>(taskData->inputs_count[0]); i++) {
        input_[i] = tmp_ptr[i];
      }
      for (int proc = 1; proc < world.size(); proc++) {
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
      return (taskData->outputs_count[0] == 1 && !(taskData->inputs.empty()));
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
  T res{};
  int n{};
  int m{};
  boost::mpi::communicator world;
};

template class chistov_a_sum_of_matrix_elements::TestMPITaskSequential<int>;
template class chistov_a_sum_of_matrix_elements::TestMPITaskSequential<double>;
template class chistov_a_sum_of_matrix_elements::TestMPITaskParallel<int>;
template class chistov_a_sum_of_matrix_elements::TestMPITaskParallel<double>;
}  // namespace chistov_a_sum_of_matrix_elements