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

int makeLinCoords(int x, int y, int xSize);

std::vector<double> genElementaryMatrix(int rows, int columns);

namespace drozhdinov_d_gauss_vertical_scheme_mpi {

std::vector<int> getRandomVector(int sz);

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
  double _elem{};
  std::vector<double> _coefs;
  std::vector<double> _b;
  std::vector<double> _x;
  boost::mpi::communicator world;
  std::vector<double> GaussVeticalScheme(const std::vector<double>& matrix, int rows, int cols,
                                    const std::vector<double>& vec);
};

}  // namespace drozhdinov_d_gauss_vertical_scheme_mpi