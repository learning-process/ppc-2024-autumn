// Filatev Vladislav Metod Zedela
#pragma once

#include <algorithm>
#include <iostream>
#include <vector>

#include "core/task/include/task.hpp"

namespace filatev_v_metod_zedela_seq {

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
  int size;
  double alfa;
  std::vector<int> matrix;
  std::vector<int> bVectrot;
  std::vector<double> answer;
  std::vector<int> delit;
};

}  // namespace filatev_v_metod_zedela_seq
