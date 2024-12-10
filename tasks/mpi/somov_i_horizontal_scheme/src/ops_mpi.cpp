#include "mpi/somov_i_horizontal_scheme/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

void somov_i_horizontal_scheme::distribute_matrix_rows(int32_t row, int32_t col, int32_t numProc,
                                                       std::vector<int32_t>& matrix_sizes,
                                                       std::vector<int32_t>& peremeshcheniye_s) {
  matrix_sizes.resize(numProc, 0);
  peremeshcheniye_s.resize(numProc, 0);

  if (numProc >= row) {
    for (int32_t i = 0; i < row; ++i) {
      matrix_sizes[i] = col;
      peremeshcheniye_s[i] = i * col;
    }
  } else {
    int32_t baseRowsPerProc = row / numProc;
    int32_t remainingRows = row % numProc;

    int32_t offset = 0;
    for (int32_t i = 0; i < numProc; ++i) {
      matrix_sizes[i] = baseRowsPerProc * col;
      if (remainingRows > 0) {
        matrix_sizes[i] += col;
        --remainingRows;
      }

      peremeshcheniye_s[i] = offset;
      offset += matrix_sizes[i];
    }
  }
}

bool somov_i_horizontal_scheme::MatrixVectorTaskMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    bool validMatrix = taskData->inputs[0] != nullptr && taskData->inputs_count[0] > 0;
    bool validVector = taskData->inputs[1] != nullptr && taskData->inputs_count[1] > 0;
    bool validDimensions = validMatrix && validVector && taskData->inputs_count[0] % taskData->inputs_count[1] == 0;
    bool validResult =
        validDimensions && taskData->outputs_count[0] == taskData->inputs_count[0] / taskData->inputs_count[1];

    if (validResult) {
      std::vector<int32_t> validSizes;
      std::vector<int32_t> validDispls;
      size_t validCols = taskData->inputs_count[1];
      size_t validRows = taskData->inputs_count[0] / taskData->inputs_count[1];
      size_t validNumProc = world.size();
      somov_i_horizontal_scheme::distribute_matrix_rows(validRows, validCols, validNumProc, validSizes, validDispls);

      bool sizesEqNumProc = validSizes.size() == validNumProc;
      bool displsEqNumProc = validDispls.size() == validNumProc;
      size_t i;
      for (i = 0; i < validNumProc; ++i) {
        if (i < validRows) {
          if (validSizes[i] == static_cast<int32_t>(validCols)) break;
          if (validDispls[i] == static_cast<int32_t>(i * validCols)) break;
        } else {
          if (validSizes[i] == 0) break;
          if (validDispls[i] == 0) break;
        }
      }
      bool flag = i == validNumProc - 1;
      return sizesEqNumProc && displsEqNumProc && flag;
    }
  }
  return true;
}

bool somov_i_horizontal_scheme::MatrixVectorTaskMPI::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* matrixData = reinterpret_cast<int32_t*>(taskData->inputs[0]);
    int32_t matrixSize = taskData->inputs_count[0];
    auto* vectorData = reinterpret_cast<int32_t*>(taskData->inputs[1]);
    int32_t vectorSize = taskData->inputs_count[1];

    matrix_.assign(matrixData, matrixData + matrixSize);
    vector_.assign(vectorData, vectorData + vectorSize);
    colCount_ = vector_.size();
    rowCount_ = matrix_.size() / colCount_;

    int32_t resultSize = taskData->outputs_count[0];
    result_.resize(resultSize, 0);

    distribute_matrix_rows(rowCount_, colCount_, world.size(), raspredeleniye, peremeshcheniye);
  }
  return true;
}

bool somov_i_horizontal_scheme::MatrixVectorTaskMPI::run() {
  internal_order_test();
  boost::mpi::broadcast(world, colCount_, 0);
  boost::mpi::broadcast(world, vector_, 0);
  boost::mpi::broadcast(world, raspredeleniye, 0);

  int32_t localNumElements = raspredeleniye[world.rank()];
  int32_t localNumRows = localNumElements / colCount_;
  std::vector<int32_t> localMatrix(localNumElements);

  if (world.rank() == 0) {
    boost::mpi::scatterv(world, matrix_.data(), raspredeleniye, peremeshcheniye, localMatrix.data(), localNumElements,
                         0);
  } else {
    boost::mpi::scatterv(world, localMatrix.data(), localNumElements, 0);
  }

  std::vector<int32_t> localResult(localNumRows, 0);
  for (int32_t i = 0; i < localNumRows; ++i) {
    for (int32_t j = 0; j < colCount_; ++j) {
      localResult[i] += localMatrix[i * colCount_ + j] * vector_[j];
    }
  }

  std::vector<int32_t> gatherCounts;
  std::vector<int32_t> gatherDisplacements;

  if (world.rank() == 0) {
    gatherCounts.resize(world.size());
    gatherDisplacements.resize(world.size());
    for (int32_t i = 0; i < world.size(); ++i) {
      int32_t numElements = raspredeleniye[i] / colCount_;
      gatherCounts[i] = numElements;
    }

    gatherDisplacements[0] = 0;
    for (int32_t i = 1; i < world.size(); ++i) {
      gatherDisplacements[i] = gatherDisplacements[i - 1] + gatherCounts[i - 1];
    }
  }

  if (world.rank() == 0) {
    boost::mpi::gatherv(world, localResult.data(), localResult.size(), result_.data(), gatherCounts,
                        gatherDisplacements, 0);
  } else {
    boost::mpi::gatherv(world, localResult.data(), localResult.size(), 0);
  }

  return true;
}

bool somov_i_horizontal_scheme::MatrixVectorTaskMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* outputData = reinterpret_cast<int32_t*>(taskData->outputs[0]);
    std::copy(result_.begin(), result_.end(), outputData);
  }
  return true;
}

bool somov_i_horizontal_scheme::MatrixVectorTask::validation() {
  internal_order_test();
  bool validMatrix = taskData->inputs_count[0] > 0;
  bool validVector = taskData->inputs_count[1] > 0;
  bool validDimensions = validMatrix && validVector && taskData->inputs_count[0] % taskData->inputs_count[1] == 0;
  bool validResult =
      validDimensions && taskData->outputs_count[0] == taskData->inputs_count[0] / taskData->inputs_count[1];

  return validResult;
}

bool somov_i_horizontal_scheme::MatrixVectorTask::pre_processing() {
  internal_order_test();
  auto* matrixData = reinterpret_cast<int32_t*>(taskData->inputs[0]);
  int32_t matrixSize = taskData->inputs_count[0];
  auto* vectorData = reinterpret_cast<int32_t*>(taskData->inputs[1]);
  int32_t vectorSize = taskData->inputs_count[1];

  matrix_.assign(matrixData, matrixData + matrixSize);
  vector_.assign(vectorData, vectorData + vectorSize);
  colCount_ = vector_.size();
  rowCount_ = matrix_.size() / colCount_;

  int32_t resultSize = taskData->outputs_count[0];
  result_.resize(resultSize, 0);

  return true;
}

bool somov_i_horizontal_scheme::MatrixVectorTask::run() {
  internal_order_test();
  for (int32_t i = 0; i < rowCount_; i++) {
    int32_t rowOff = i * colCount_;
    int32_t sum = 0;
    for (int32_t j = 0; j < colCount_; j++) {
      sum += matrix_[rowOff + j] * vector_[j];
    }
    result_[i] = sum;
  }
  return true;
}

bool somov_i_horizontal_scheme::MatrixVectorTask::post_processing() {
  internal_order_test();
  auto* outputData = reinterpret_cast<int32_t*>(taskData->outputs[0]);
  std::copy(result_.begin(), result_.end(), outputData);
  return true;
}
