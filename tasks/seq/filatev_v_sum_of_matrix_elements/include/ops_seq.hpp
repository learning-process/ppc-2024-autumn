// Filatev Vladislav Sum_of_matrix_elements
#pragma once

#include <string>
#include <random>
#include <vector>
#include "core/task/include/task.hpp"

namespace filatev_v_sum_of_matrix_elements_seq {

  long long sumVector(std::vector<int> vector);
  std::vector<std::vector<int>> getRandomMatrix(int size_n, int size_m);

class SumMatrix : public ppc::core::Task {
 public:
  explicit SumMatrix(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> matrix;
  long long summ = 0;
  int size_n, size_m;
};

}  // filatev_v_sum_of_matrix_elements_seq