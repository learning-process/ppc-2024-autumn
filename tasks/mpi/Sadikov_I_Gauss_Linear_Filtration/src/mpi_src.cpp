#include "mpi/Sadikov_I_Gauss_Linear_Filtration/include/ops_mpi.h"

Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationSeq::LinearFiltrationSeq(
    std::shared_ptr<ppc::core::TaskData> taskData)
    : Task(std::move(taskData)) {}

bool Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationSeq::validation() {
  internal_order_test();
  if ((taskData->inputs_count[0] > 2 || taskData->inputs_count[0] == 0) &&
      (taskData->inputs_count[1] > 2 || taskData->inputs_count[1] == 0)) {
    return taskData->outputs_count[0] == taskData->inputs_count[0] * taskData->inputs_count[1];
  }
  return false;
}

bool Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationSeq::pre_processing() {
  internal_order_test();
  m_rowsCount = static_cast<int>(taskData->inputs_count[0]);
  m_columnsCount = static_cast<int>(taskData->inputs_count[1]);
  m_pixelsMatrix.reserve(m_rowsCount * m_columnsCount);
  auto *tmpPtr = reinterpret_cast<Point<double> *>(taskData->inputs[0]);
  for (int i = 0; i < m_columnsCount * m_rowsCount; ++i) {
    m_pixelsMatrix.emplace_back(tmpPtr[i]);
  }
  m_gaussMatrix.reserve(9);
  CalculateGaussMatrix();
  m_outMatrix = std::vector<Point<double>>(m_columnsCount * m_rowsCount);
  return true;
}

bool Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationSeq::run() {
  internal_order_test();
  for (auto i = 0; i < m_rowsCount; ++i) {
    for (auto j = 0; j < m_columnsCount; ++j) {
      CalculateNewPixelValue(i, j);
    }
  }
  return true;
}

bool Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationSeq::post_processing() {
  internal_order_test();
  for (auto i = 0; i < m_rowsCount * m_columnsCount; ++i) {
    reinterpret_cast<Point<double> *>(taskData->outputs[0])[i] = m_outMatrix[i];
  }
  return true;
}

void Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationSeq::CalculateGaussMatrix() {
  double sum = 0;
  for (auto x = -1; x < 2; ++x) {
    for (auto y = -1; y < 2; ++y) {
      m_gaussMatrix.emplace_back(1 / (2 * std::numbers::pi * sigma * sigma) *
                                 std::pow(std::numbers::e, -((x * x + y * y) / 2.0 * sigma * sigma)));
      sum += m_gaussMatrix.back();
    }
  }
  for (auto &&it : m_gaussMatrix) {
    it /= sum;
  }
}

void Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationSeq::CalculateNewPixelValue(int iIndex, int jIndex) {
  auto index = 0;
  for (auto g = 0; g < 9; ++g) {
    if (g < 3) {
      if (CheckRowIndex(iIndex - 1) and CheckColumnIndex(jIndex - 1 + g)) {
        index = (iIndex - 1) * m_columnsCount + jIndex - 1 + g;
        if (CheckIndex(index)) {
          m_outMatrix[iIndex * m_columnsCount + jIndex] += m_pixelsMatrix[index] * m_gaussMatrix[g];
        }
      }
    } else if (g > 2 && g < 6) {
      index = iIndex * m_columnsCount + jIndex - 4 + g;
      if (CheckColumnIndex(jIndex - 4 + g)) {
        if (CheckIndex(index)) {
          m_outMatrix[iIndex * m_columnsCount + jIndex] += m_pixelsMatrix[index] * m_gaussMatrix[g];
        }
      }
    } else {
      if (CheckRowIndex(iIndex + 1) and CheckColumnIndex(jIndex - 7 + g)) {
        index = (iIndex + 1) * m_columnsCount + jIndex - 7 + g;
        if (CheckIndex(index)) {
          m_outMatrix[iIndex * m_columnsCount + jIndex] += m_pixelsMatrix[index] * m_gaussMatrix[g];
        }
      }
    }
  }
}

Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationMPI::LinearFiltrationMPI(
    std::shared_ptr<ppc::core::TaskData> taskData)
    : Task(std::move(taskData)) {}

bool Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if ((taskData->inputs_count[0] > 2 || taskData->inputs_count[0] == 0) &&
        (taskData->inputs_count[1] > 2 || taskData->inputs_count[1] == 0)) {
      return taskData->outputs_count[0] == taskData->inputs_count[0] * taskData->inputs_count[1];
    }
    return false;
  }
  return true;
}

bool Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationMPI::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    m_rowsCount = static_cast<int>(taskData->inputs_count[0]);
    m_columnsCount = static_cast<int>(taskData->inputs_count[1]);
    m_pixelsMatrix.reserve(m_rowsCount * m_columnsCount);
    auto *tmpPtr = reinterpret_cast<Point<double> *>(taskData->inputs[0]);
    for (auto column = 0; column < m_columnsCount; ++column) {
      for (auto row = 0; row < m_rowsCount; ++row) {
        m_pixelsMatrix.emplace_back(tmpPtr[m_columnsCount * row + column]);
      }
    }
    const int proceses = m_columnsCount / m_minColumnsCount;
    if (proceses > world.size()) {
      int averegeCount = m_columnsCount / world.size();
      m_sizes = std::vector<int>(world.size(), averegeCount * m_rowsCount);
      m_sizes.back() += (m_columnsCount % world.size()) * m_rowsCount;
    } else {
      int m_columns = m_columnsCount;
      if (!m_pixelsMatrix.empty()) {
        m_sizes.resize(world.size());
      }
      for (auto i = 0; i < world.size(); ++i) {
        if (m_columns >= m_minColumnsCount) {
          m_sizes[i] = m_minColumnsCount * m_rowsCount;
          m_columns -= m_minColumnsCount;
          if (m_columns < m_minColumnsCount) {
            m_sizes[i] += m_columns * m_rowsCount;
            break;
          }
        }
      }
    }
    m_gaussMatrix.reserve(9);
    m_outMatrix = std::vector<Point<double>>(m_columnsCount * m_rowsCount);
  }
  CalculateGaussMatrix();
  return true;
}

bool Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationMPI::run() {
  internal_order_test();
  broadcast(world, m_sizes, 0);
  broadcast(world, m_columnsCount, 0);
  broadcast(world, m_rowsCount, 0);
  broadcast(world, m_pixelsMatrix, 0);
  if (world.rank() < static_cast<int>(m_sizes.size())) {
    m_intermediateRes.resize(m_sizes[world.rank()]);
    int position = 0;
    for (auto row = std::accumulate(m_sizes.begin(), m_sizes.begin() + world.rank(), 0) / m_rowsCount;
         row < std::accumulate(m_sizes.begin(), m_sizes.begin() + world.rank() + 1, 0) / m_rowsCount; ++row) {
      for (auto column = 0; column < m_rowsCount; ++column) {
        CalculateNewPixelValue(row, column, position);
        position++;
      }
    }
  }
  if (world.rank() == 0 && !m_pixelsMatrix.empty()) {
    std::vector<Point<double>> localRes(m_rowsCount * m_columnsCount);
    boost::mpi::gatherv(world, m_intermediateRes, localRes.data(), m_sizes, 0);
    m_outMatrix = std::move(localRes);
  } else if (world.rank() != 0 && !m_pixelsMatrix.empty()) {
    boost::mpi::gatherv(world, m_intermediateRes, 0);
  }
  return true;
}

bool Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (auto i = 0; i < m_rowsCount; ++i) {
      for (auto j = 0; j < m_columnsCount; ++j) {
        reinterpret_cast<Point<double> *>(taskData->outputs[0])[j + i * m_columnsCount] =
            m_outMatrix[m_rowsCount * j + i];
      }
    }
  }
  return true;
}

void Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationMPI::CalculateGaussMatrix() {
  double sum = 0;
  for (auto x = -1; x < 2; ++x) {
    for (auto y = -1; y < 2; ++y) {
      m_gaussMatrix.emplace_back(1 / (2 * std::numbers::pi * sigma * sigma) *
                                 std::pow(std::numbers::e, -((x * x + y * y) / 2.0 * sigma * sigma)));
      sum += m_gaussMatrix.back();
    }
  }
  for (auto &&it : m_gaussMatrix) {
    it /= sum;
  }
}

void Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationMPI::CalculateNewPixelValue(int iIndex, int jIndex,
                                                                                    int position) {
  auto index = 0;

  for (auto g = 0; g < 9; ++g) {
    if (g < 3) {
      if (CheckRowIndex(jIndex - 1 + g) and CheckColumnIndex(iIndex - 1)) {
        index = (iIndex - 1) * m_rowsCount + jIndex - 1 + g;
        if (CheckIndex(index)) {
          m_intermediateRes[position] += m_pixelsMatrix[index] * m_gaussMatrix[g];
        }
      }
    } else if (g > 2 && g < 6) {
      index = iIndex * m_rowsCount + jIndex - 4 + g;
      if (CheckRowIndex(jIndex - 4 + g)) {
        if (CheckIndex(index)) {
          m_intermediateRes[position] += m_pixelsMatrix[index] * m_gaussMatrix[g];
        }
      }
    } else {
      if (CheckRowIndex(jIndex - 7 + g) and CheckColumnIndex(iIndex + 1)) {
        index = (iIndex + 1) * m_rowsCount + jIndex - 7 + g;
        if (CheckIndex(index)) {
          m_intermediateRes[position] += m_pixelsMatrix[index] * m_gaussMatrix[g];
        }
      }
    }
  }
}