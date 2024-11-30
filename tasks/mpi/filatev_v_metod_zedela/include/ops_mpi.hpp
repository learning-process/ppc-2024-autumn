// Filatev Vladislav Metod Zedela
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

namespace filatev_v_metod_zedela_mpi {

class MetodZedela : public ppc::core::Task {
 public:
  explicit MetodZedela(std::shared_ptr<ppc::core::TaskData> taskData_);
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void setAlfa(double alfa);
  int rankMatrix(std::vector<int>& matrixT, int n) const;
  int rankRMatrix();
  double determinant();

 private:
  boost::mpi::communicator world;
  boost::mpi::status status;
  int size;
  double alfa;
  std::vector<int> matrix;
  std::vector<int> tMatrix;
  std::vector<int> bVectrot;
  std::vector<double> answer;
  std::vector<int> delit;
};

class TestClassForMetodZedela {
 public:
  int generatorVector(std::vector<int>& vec);
  void generatorMatrix(std::vector<int>& matrix, int size);
  void genetatirVectorB(std::vector<int>& matrix, std::vector<int>& vecB);
  bool rightAns(std::vector<double>& ans, double alfa);

 private:
  std::vector<int> ans;
};

}  // namespace filatev_v_metod_zedela_mpi