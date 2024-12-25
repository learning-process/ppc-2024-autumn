#include "seq/sedova_o_multiplication_matrices_ccs/include/ops_seq.hpp"

bool sedova_o_multiplication_matrices_ccs_seq::MatrixMultiplicationCCS::validation() {
  internal_order_test();
  // Retrieve matrix dimensions from task data
  int rows_A = *reinterpret_cast<int*>(taskData->inputs[0]);
  int cols_A = *reinterpret_cast<int*>(taskData->inputs[1]);
  int rows_B = *reinterpret_cast<int*>(taskData->inputs[2]);
  int cols_B = *reinterpret_cast<int*>(taskData->inputs[3]);

  // Validate dimensions: all must be positive and cols of A must match rows of B
  if (rows_A <= 0 || cols_A <= 0 || rows_B <= 0 || cols_B <= 0) {
    return false;  // Invalid dimensions
  }

  if (cols_A != rows_B) {
    return false;  // Incompatible for multiplication
  }

  return true;
};

bool sedova_o_multiplication_matrices_ccs_seq::MatrixMultiplicationCCS::pre_processing() {
  internal_order_test();

  // Extract matrix dimensions from task data
  rowsA = *reinterpret_cast<int*>(taskData->inputs[0]);
  colsA = *reinterpret_cast<int*>(taskData->inputs[1]);
  rowsB = *reinterpret_cast<int*>(taskData->inputs[2]);
  colsB = *reinterpret_cast<int*>(taskData->inputs[3]);

    // Load matrix A values, row and column
    int ccsValuesCountA = taskData->inputs_count[4];
    auto* ccsValuesPtrA = reinterpret_cast<double*>(taskData->inputs[4]);
    int ccsRowsIndCountA = taskData->inputs_count[5];
    auto* ccsRowsIndPtrA = reinterpret_cast<int*>(taskData->inputs[5]);
    int ccsColsCountA = taskData->inputs_count[6];
    auto* ccsColsPtrA = reinterpret_cast<int*>(taskData->inputs[6]);
    ccsValuesA.assign(ccsValuesPtrA, ccsValuesPtrA + ccsValuesCountA);
    ccsRowIndA.assign(ccsRowsIndPtrA, ccsRowsIndPtrA + ccsRowsIndCountA);
    ccsColPtrA.assign(ccsColsPtrA, ccsColsPtrA + ccsColsCountA);

    // Load matrix B values, row and column
    int ccsValuesCountB = taskData->inputs_count[7];
    auto* ccsValuesPtrB = reinterpret_cast<double*>(taskData->inputs[7]);
    int ccsRowsIndCountB = taskData->inputs_count[8];
    auto* ccsRowsIndPtrB = reinterpret_cast<int*>(taskData->inputs[8]);
    int ccsColsCountB = taskData->inputs_count[9];
    auto* ccsColsPtrB = reinterpret_cast<int*>(taskData->inputs[9]);
    ccsValuesB.assign(ccsValuesPtrB, ccsValuesPtrB + ccsValuesCountB);
    ccsRowIndB.assign(ccsRowsIndPtrB, ccsRowsIndPtrB + ccsRowsIndCountB);
    ccsColPtrB.assign(ccsColsPtrB, ccsColsPtrB + ccsColsCountB);

    // Transpose matrix A to prepare for multiplication
    transposeCCS(ccsValuesA, ccsRowIndA, ccsColPtrA, rowsA, colsA, outputValues, outputRowInd, outputColPtr);

    // Update transposed dimensions
    rowsC = colsA;
    colsC = rowsA;
  return true;
};

bool sedova_o_multiplication_matrices_ccs_seq::MatrixMultiplicationCCS::run() {
  internal_order_test();
  performCCSMultiplication(outputValues, outputRowInd, outputColPtr, rowsC, ccsValuesB, ccsRowIndB, ccsColPtrB, colsB, ResultValues, ResultInd, ResultPtr);
  return true;
};

bool sedova_o_multiplication_matrices_ccs_seq::MatrixMultiplicationCCS::post_processing() {
  internal_order_test();
    // Retrieve pointers to output data arrays
    auto* outputValuesPtr = reinterpret_cast<double*>(taskData->outputs[0]);
    auto* outputRowIndPtr = reinterpret_cast<int*>(taskData->outputs[1]);
    auto* outputColsPtr = reinterpret_cast<int*>(taskData->outputs[2]);

    // Copy results into the output arrays
    std::copy(ResultValues.begin(), ResultValues.end(), outputValuesPtr);
    std::copy(ResultInd.begin(), ResultInd.end(), outputRowIndPtr);
    std::copy(ResultPtr.begin(), ResultPtr.end(), outputColsPtr);

  return true;
};