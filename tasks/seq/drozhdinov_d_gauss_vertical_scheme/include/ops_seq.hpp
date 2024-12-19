// Copyright 2023 Nesterov Alexander
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

#define GAMMA 1e-9

int mkLinCoordddm(int x, int y, int xSize);
double myrnd(double value);

namespace drozhdinov_d_gauss_vertical_scheme_seq {
int Myrank(std::vector<double> matrix, int m, int n);
std::vector<double> extendedMatrix(const std::vector<double>& A, int n, const std::vector<double>& b);
double Determinant(const std::vector<double>& matrix, int n);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int rows{}, columns{};
  double elem{};
  std::vector<double> coefs;
  std::vector<double> b;
  std::vector<double> x;
  std::vector<int> row_number;
  std::vector<bool> major;
};

}  // namespace drozhdinov_d_gauss_vertical_scheme_seq