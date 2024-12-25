#pragma once
#include <gtest/gtest.h>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <memory>
#include <vector>
#include <utility>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <unordered_map>
#include "core/task/include/task.hpp"

namespace sedova_o_multiplication_matrices_ccs_mpi {

inline void convertMatrixToCCS(const std::vector<std::vector<double>>& inputMatrix, int rowCount, int colCount,
                               std::vector<double>& ccsValues, std::vector<int>& ccsRowIndices,
                               std::vector<int>& ccsColPtr) {
  // Clear the output vectors to avoid accumulating data from previous calls
  ccsValues.clear();
  ccsRowIndices.clear();
  ccsColPtr.clear();

  // Initialize the starting point for the first column
  ccsColPtr.push_back(0);

  // Iterate over each column of the input matrix
  for (int col = 0; col < colCount; ++col) {
    // Iterate over each row in the current column
    for (int row = 0; row < rowCount; ++row) {
      // If the element is not zero, add it to the CCS representation
      if (inputMatrix[row][col] != 0.0) {
        ccsValues.push_back(inputMatrix[row][col]);  // Add the non-zero value
        ccsRowIndices.push_back(row);                // Add the corresponding row index
      }
    }
    // Update the pointer to the start of the next column
    ccsColPtr.push_back(ccsValues.size());
  }
}


inline void transposeCCS(const std::vector<double>& inputValues, const std::vector<int>& inputRowIndices,
                         const std::vector<int>& inputColPtr, int totalRows, int totalCols,
                         std::vector<double>& outputValues, std::vector<int>& outputRowIndices,
                         std::vector<int>& outputColPtr) {
  // Initialize the output column pointer with zero for the first column
  outputColPtr.resize(totalRows + 1, 0);  // We need one extra for the last column pointer

  // Count non-zero entries in each row of the transposed matrix
  for (int col = 0; col < totalCols; ++col) {
    for (int index = inputColPtr[col]; index < inputColPtr[col + 1]; ++index) {
      int rowIndex = inputRowIndices[index];  // Original row index
      outputColPtr[rowIndex + 1]++;           // Increment count for this row in transposed matrix
    }
  }

  // Convert counts to actual column pointers
  for (int i = 1; i <= totalRows; ++i) {
    outputColPtr[i] += outputColPtr[i - 1];  // Cumulative sum to get column pointers
  }

  // Resize output vectors based on the number of non-zero entries
  outputValues.resize(outputColPtr[totalRows]);
  outputRowIndices.resize(outputColPtr[totalRows]);

  // Fill in the transposed values and row indices
  std::vector<int> currentIndex(totalRows, 0);  // Track current index for each row
  for (int col = 0; col < totalCols; ++col) {
    for (int index = inputColPtr[col]; index < inputColPtr[col + 1]; ++index) {
      int rowIndex = inputRowIndices[index];                            // Original row index
      int insertPos = outputColPtr[rowIndex] + currentIndex[rowIndex];  // Position to insert

      // Place value and corresponding row index in the output vectors
      outputValues[insertPos] = inputValues[index];
      outputRowIndices[insertPos] = col;

      currentIndex[rowIndex]++;  // Move to next position for this row
    }
  }
}

inline void extractColumns(const std::vector<double>& values, const std::vector<int>& rowIndices,
                           const std::vector<int>& colPtr, int startColumn, int endColumn,
                           std::vector<double>& extractedValues, std::vector<int>& extractedRowIndices,
                           std::vector<int>& extractedColPtr) {
  // Clear output vectors to ensure they are empty before extraction
  extractedValues.clear();
  extractedRowIndices.clear();
  extractedColPtr.clear();

  // Initialize column pointer for the extracted matrix
  extractedColPtr.push_back(0);

  // Calculate total number of non-zero entries in the specified columns
  int totalNonZeroEntries = 0;
  for (int j = startColumn; j < endColumn; ++j) {
    int numEntriesInColumn = colPtr[j + 1] - colPtr[j];  // Number of non-zero entries in this column
    totalNonZeroEntries += numEntriesInColumn;           // Accumulate total count

    // Update the column pointer for the extracted matrix
    extractedColPtr.push_back(totalNonZeroEntries);
  }

  // Resize output vectors to hold all non-zero entries
  extractedValues.resize(totalNonZeroEntries);
  extractedRowIndices.resize(totalNonZeroEntries);

  // Fill in the extracted values and row indices
  int currentIndex = 0;  // Current index for filling output vectors
  for (int j = startColumn; j < endColumn; ++j) {
    for (int k = colPtr[j]; k < colPtr[j + 1]; ++k) {
      // Directly assign values and row indices to their respective positions
      extractedValues[currentIndex] = values[k];
      extractedRowIndices[currentIndex] = rowIndices[k];
      currentIndex++;  // Move to next position in output vectors
    }
  }
}

// Helper function to initialize B values for a specific column
inline void fillBValues(const std::vector<double>& values_B, const std::vector<int>& row_indices_B,
                 const std::vector<int>& col_ptr_B, int col_B, std::vector<double>& B_values,
                 std::vector<bool>& B_row_used) {
  // Reset temporary storage
  std::fill(B_values.begin(), B_values.end(), 0.0);
  std::fill(B_row_used.begin(), B_row_used.end(), false);

  // Populate B_values and mark used rows
  for (int i = col_ptr_B[col_B]; i < col_ptr_B[col_B + 1]; ++i) {
    int row_B = row_indices_B[i];
    B_values[row_B] = values_B[i];
    B_row_used[row_B] = true;
  }
}

// Helper function to compute contributions to C for a given column of A
inline void computeColumnContribution(const std::vector<double>& values_A, const std::vector<int>& row_indices_A,
                               const std::vector<int>& col_ptr_A, int col_A, const std::vector<double>& B_values,
                               const std::vector<bool>& B_row_used, double& sum, bool& non_zero) {
  sum = 0.0;         // Initialize sum for this column
  non_zero = false;  // Flag to check if we have a non-zero sum

  // Calculate the dot product for this entry in the result matrix
  for (int i = col_ptr_A[col_A]; i < col_ptr_A[col_A + 1]; ++i) {
    int row_A = row_indices_A[i];
    if (B_row_used[row_A]) {
      sum += values_A[i] * B_values[row_A];
    }
  }

  // Check if the sum is non-zero
  if (sum != 0.0) {
    non_zero = true;  // Mark that we have a valid contribution
  }
}

// Main function to perform matrix multiplication in CCS format
inline void performCCSMultiplication(const std::vector<double>& values_A, const std::vector<int>& row_indices_A,
                                     const std::vector<int>& col_ptr_A, int num_rows_A,
                                     const std::vector<double>& values_B, const std::vector<int>& row_indices_B,
                                     const std::vector<int>& col_ptr_B, int num_cols_B, std::vector<double>& values_C,
                                     std::vector<int>& row_indices_C, std::vector<int>& col_ptr_C) {
  // Clear output vectors
  values_C.clear();
  row_indices_C.clear();
  col_ptr_C.clear();
  col_ptr_C.push_back(0);  // Initialize the first column pointer

  // Temporary storage for matrix B
  std::vector<double> B_values(num_rows_A, 0.0);
  std::vector<bool> B_row_used(num_rows_A, false);

  // Iterate over each column of matrix B
  for (int col_B = 0; col_B < num_cols_B; ++col_B) {
    // Fill temporary storage for current column of B
    fillBValues(values_B, row_indices_B, col_ptr_B, col_B, B_values, B_row_used);

    // Iterate over each column of matrix A to compute contributions to C
    for (int col_A = 0; col_A < static_cast<int>(col_ptr_A.size()) - 1; ++col_A) {
      double sum;
      bool has_non_zero;

      // Compute contribution from column A to column C
      computeColumnContribution(values_A, row_indices_A, col_ptr_A, col_A, B_values, B_row_used, sum, has_non_zero);

      // If there is a non-zero contribution, store it in the result vectors
      if (has_non_zero) {
        values_C.push_back(sum);
        row_indices_C.push_back(col_A);
      }
    }

    // Update the column pointer for the result matrix
    col_ptr_C.push_back(values_C.size());
  }
}

inline std::pair<int, int> determineSegment(int totalElements, int totalProcesses, int currentRank) {
  // Calculate the number of elements each process will handle
  int totalPerProcess = (totalElements + totalProcesses - 1) / totalProcesses;  // Ceiling division

  // Calculate start index
  int startIdx = currentRank * totalPerProcess;

  // Calculate end index
  int endIdx = std::min(startIdx + totalPerProcess, totalElements);  // Ensure we don't exceed totalElements

  return {startIdx, endIdx};
}

class MatrixMultiplicationCCS : public ppc::core::Task {
 public:
  explicit MatrixMultiplicationCCS(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
 private:
  // Matrix storage
  std::vector<std::vector<double>> matrixA, matrixB;
  int rowsA, colsA, rowsB, colsB, rowsC, colsC, color, start, end, cols;

  // CCS representations
  std::vector<double> ccsValuesA, ccsValuesB;
  std::vector<int> ccsRowIndA, ccsColPtrA, ccsRowIndB, ccsColPtrB;

  // Result storage
  std::vector<double> outputValues;
  std::vector<int> outputRowInd, outputColPtr;

  std::vector<double> ResultValues;
  std::vector<int> ResultInd, ResultPtr;

  boost::mpi::communicator world, mpiComm;
};
}