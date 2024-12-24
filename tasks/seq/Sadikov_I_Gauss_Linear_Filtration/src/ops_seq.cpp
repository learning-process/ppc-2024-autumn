#include "seq/Sadikov_I_Gauss_Linear_Filtration/include/seq_task.h"

Sadikov_I_Gauss_Linear_Filtration::LinearFiltration::LinearFiltration(std::shared_ptr<ppc::core::TaskData> taskData)
    : Task(std::move(taskData)) {}

bool Sadikov_I_Gauss_Linear_Filtration::LinearFiltration::validation() {
  internal_order_test();
  if ((taskData->inputs_count[0] > 2 || taskData->inputs_count[0] == 0) &&
      (taskData->inputs_count[1] > 2 || taskData->inputs_count[1] == 0)) {
    return taskData->outputs_count[0] == taskData->inputs_count[0] * taskData->inputs_count[1];
  }
  return false;
}

bool Sadikov_I_Gauss_Linear_Filtration::LinearFiltration::pre_processing() {
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

bool Sadikov_I_Gauss_Linear_Filtration::LinearFiltration::run() {
  internal_order_test();
  for (auto i = 0; i < m_rowsCount; ++i) {
    for (auto j = 0; j < m_columnsCount; ++j) {
      CalculateNewPixelValue(i, j);
    }
  }
  return true;
}

bool Sadikov_I_Gauss_Linear_Filtration::LinearFiltration::post_processing() {
  internal_order_test();
  for (auto i = 0; i < m_rowsCount * m_columnsCount; ++i) {
    reinterpret_cast<Point<double> *>(taskData->outputs[0])[i] = m_outMatrix[i];
  }
  return true;
}

void Sadikov_I_Gauss_Linear_Filtration::LinearFiltration::CalculateGaussMatrix() {
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

void Sadikov_I_Gauss_Linear_Filtration::LinearFiltration::CalculateNewPixelValue(int iIndex, int jIndex) {
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