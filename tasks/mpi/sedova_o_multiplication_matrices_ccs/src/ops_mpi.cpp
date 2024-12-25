#include "mpi/sedova_o_multiplication_matrices_ccs/include/ops_mpi.hpp"


  bool sedova_o_multiplication_matrices_ccs_mpi::MatrixMultiplicationCCS::validation() {
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

  bool sedova_o_multiplication_matrices_ccs_mpi::MatrixMultiplicationCCS::pre_processing() {
    internal_order_test();

    // Extract matrix dimensions from task data
    rowsA = *reinterpret_cast<int*>(taskData->inputs[0]);
    colsA = *reinterpret_cast<int*>(taskData->inputs[1]);
    rowsB = *reinterpret_cast<int*>(taskData->inputs[2]);
    colsB = *reinterpret_cast<int*>(taskData->inputs[3]);

    // Only the root process (rank 0) will load the matrix data
    if (world.rank() == 0) {
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
    }

    return true;
  };

bool sedova_o_multiplication_matrices_ccs_mpi::MatrixMultiplicationCCS::run() {
  internal_order_test();
  std::vector<double> Values, Result;
  std::vector<int> rowInd, colPtr, ResultRowInd, ResultColPtr;
  // Determine communication color based on rank
  color = static_cast<int>(world.rank() < colsB);
  mpiComm = world.split(color);
  try {
    boost::mpi::broadcast(mpiComm, ccsValuesB, 0);
    boost::mpi::broadcast(mpiComm, ccsRowIndB, 0);
    boost::mpi::broadcast(mpiComm, ccsColPtrB, 0);

    boost::mpi::broadcast(mpiComm, outputValues, 0);
    boost::mpi::broadcast(mpiComm, outputRowInd, 0);
    boost::mpi::broadcast(mpiComm, outputColPtr, 0);
    boost::mpi::broadcast(mpiComm, rowsC, 0);
  } catch (const std::exception& e) {
    std::cerr << "Error during MPI broadcast: " << e.what() << std::endl;
    return false;
  }
  if (color == 1) {
    // Split workload among processes for matrix B columns
    auto segment = determineSegment(colsB, mpiComm.size(), mpiComm.rank());

    start = segment.first;
    end = segment.second;
    cols = end - start;

    // Extract relevant columns from matrix B for local processing
    try {
      extractColumns(ccsValuesB, ccsRowIndB, ccsColPtrB, start, end, Values, rowInd, colPtr);
    } catch (const std::exception& e) {
      std::cerr << "Error during column extraction: " << e.what() << std::endl;
      return false;  // Indicate failure
    }

    // Perform multiplication of transposed A with extracted B columns
    try {
      performCCSMultiplication(outputValues, outputRowInd, outputColPtr, rowsC, Values, rowInd, colPtr, cols, Result,
                               ResultRowInd, ResultColPtr);
    } catch (const std::exception& e) {
      std::cerr << "Error during CCS multiplication: " << e.what() << std::endl;
      return false;
    }

    // Gather sizes of results from each process
    std::vector<int> resultSizes(mpiComm.size(), 0);

    try {
      boost::mpi::gather(mpiComm, ResultColPtr.back(), resultSizes.data(), 0);

      int totalPointers = 0;
      boost::mpi::reduce(mpiComm, static_cast<int>(ResultColPtr.size() - 1), totalPointers, std::plus<>(), 0);

      std::vector<int> pointerSizes;
      if (mpiComm.rank() == 0) {
        pointerSizes.resize(totalPointers);
      }

      boost::mpi::gather(mpiComm, static_cast<int>(ResultColPtr.size() - 1), pointerSizes, 0);

      if (mpiComm.rank() == 0) {
        int totalResults = std::accumulate(resultSizes.begin(), resultSizes.end(), 0);
        ResultValues.resize(totalResults);
        ResultInd.resize(totalResults);
        ResultPtr.resize(totalPointers);

        // Gather results from all processes
        boost::mpi::gatherv(mpiComm, Result.data(), Result.size(), ResultValues.data(), resultSizes, 0);
        boost::mpi::gatherv(mpiComm, ResultRowInd.data(), ResultRowInd.size(), ResultInd.data(), resultSizes, 0);
        boost::mpi::gatherv(mpiComm, ResultColPtr.data(), ResultColPtr.size() - 1, ResultPtr.data(), pointerSizes, 0);

        // Adjust column pointers for gathered results
        int cumulativeShift = 0;
        int currentOffset = 0;
        for (size_t j = 0; j < pointerSizes.size(); j++) {
          cumulativeShift = resultSizes[j];
          currentOffset += pointerSizes[j];
          for (size_t i = currentOffset; i < ResultPtr.size(); i++) {
            ResultPtr[i] += cumulativeShift;
          }
        }

        ResultPtr.push_back(totalResults);  // Append total count of non-zero entries
      } else {
        // Non-root processes send their results to the root
        boost::mpi::gatherv(mpiComm, Result.data(), Result.size(), 0);        // No need to gather to rank != 0
        boost::mpi::gatherv(mpiComm, Result.data(), ResultRowInd.size(), 0);  // No need to gather to rank != 0
        boost::mpi::gatherv(mpiComm, ResultColPtr.data(), ResultColPtr.size() - 1, 0);  // No need to gather to rank != 0
      }
    } catch (const std::exception& e) {
      std::cerr << "Error during MPI gather/reduce: " << e.what() << std::endl;
      return false;
    }
  }

  return true;
};

bool sedova_o_multiplication_matrices_ccs_mpi::MatrixMultiplicationCCS::post_processing() {
  internal_order_test();

  // Only the root process (rank 0) will handle the output
  if (color == 1 && mpiComm.rank() == 0) {
    // Retrieve pointers to output data arrays
    auto* outputValuesPtr = reinterpret_cast<double*>(taskData->outputs[0]);
    auto* outputRowIndPtr = reinterpret_cast<int*>(taskData->outputs[1]);
    auto* outputColsPtr = reinterpret_cast<int*>(taskData->outputs[2]);

    // Copy results into the output arrays
    std::copy(ResultValues.begin(), ResultValues.end(), outputValuesPtr);
    std::copy(ResultInd.begin(), ResultInd.end(), outputRowIndPtr);
    std::copy(ResultPtr.begin(), ResultPtr.end(), outputColsPtr);
  }

  return true;
};