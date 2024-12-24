#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/Sadikov_I_Gauss_Linear_Filtration/include/Point.h"

namespace Sadikov_I_Gauss_Linear_Filtration {
class LinearFiltrationSeq : public ppc::core::Task {
  std::vector<Point<double>> m_pixelsMatrix;
  std::vector<Point<double>> m_outMatrix;
  std::vector<double> m_gaussMatrix;
  int m_columnsCount = 0;
  int m_rowsCount = 0;
  static constexpr double sigma = 1.0;

 public:
  explicit LinearFiltrationSeq(std::shared_ptr<ppc::core::TaskData> taskData);
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;
  void CalculateGaussMatrix();
  void CalculateNewPixelValue(int iIndex, int jIndex);
  bool CheckIndex(int index) const noexcept { return index >= 0 and index < static_cast<int>(m_pixelsMatrix.size()); }
  bool CheckRowIndex(int index) const noexcept { return index >= 0 and index < m_rowsCount; }
  bool CheckColumnIndex(int index) const noexcept { return index >= 0 and index < m_columnsCount; }
};

class LinearFiltrationMPI : public ppc::core::Task {
  std::vector<Point<double>> m_pixelsMatrix;
  std::vector<Point<double>> m_outMatrix;
  std::vector<double> m_gaussMatrix;
  int m_columnsCount = 0;
  int m_rowsCount = 0;
  static constexpr double sigma = 1.0;
  static constexpr int m_minColumnsCount = 3;
  std::vector<int> m_sizes;
  boost::mpi::communicator world;
  std::vector<Point<double>> m_intermediateRes;

 public:
  explicit LinearFiltrationMPI(std::shared_ptr<ppc::core::TaskData> taskData);
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;
  void CalculateGaussMatrix();
  void CalculateNewPixelValue(int iIndex, int jIndex, int position);
  bool CheckIndex(int index) const noexcept { return index >= 0 and index < m_rowsCount * m_columnsCount; }
  bool CheckRowIndex(int index) const noexcept { return index >= 0 and index < m_rowsCount; }
  bool CheckColumnIndex(int index) const noexcept { return index >= 0 and index < m_columnsCount; }
};
}  // namespace Sadikov_I_Gauss_Linear_Filtration
