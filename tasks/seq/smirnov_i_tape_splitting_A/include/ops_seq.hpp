#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace smirnov_i_tape_splitting_A {

void get_random_matrix(double* matr, int size);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  ~TestMPITaskSequential() override {
    delete[] A;
    delete[] B;
    delete[] res;
  }

 private:
  int m_a;
  int n_a;
  int m_b;
  int n_b;
  double* A;
  double* B;
  double* res;
};
}  // namespace smirnov_i_tape_splitting_A
