// Copyright 2023 Nesterov Alexander
#pragma once
// not example
#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

int mkLinCoordddm(int x, int y, int xSize);
std::vector<double> genDenseMatrix(int n, int a);
std::vector<double> genElementaryMatrix(int rows, int columns);
double myrnd(double value);

namespace drozhdinov_d_gauss_vertical_scheme_mpi {

template <typename T>
std::vector<T> getRandomVector(int sz);
int Myrank(std::vector<double> matrix, int m, int n);
std::vector<double> extendedMatrix(std::vector<double> A, int n, std::vector<double> b);
double Determinant(const std::vector<double>& matrix, int n);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
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

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int _rows{}, _columns{};
  std::vector<double> _coefs;
  std::vector<double> _b;
  std::vector<double> _x;
  boost::mpi::communicator world;
  std::vector<double> GaussVerticalScheme(const std::vector<double>& matrix, int rows, int cols,
                                          const std::vector<double>& vec);
};

}  // namespace drozhdinov_d_gauss_vertical_scheme_mpi