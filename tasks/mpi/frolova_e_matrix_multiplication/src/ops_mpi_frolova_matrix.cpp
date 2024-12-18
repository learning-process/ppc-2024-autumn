// Copyright 2023 Nesterov Alexander
#include "mpi/frolova_e_matrix_multiplication/include/ops_mpi_frolova_matrix.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

std::vector<int> frolova_e_matrix_multiplication_mpi::Multiplication(size_t M, size_t N, size_t K,
                                                                     const std::vector<int>& A,
                                                                     const std::vector<int>& B) {
  std::vector<int> C(M * N);

  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      C[i * N + j] = 0;

      for (size_t k = 0; k < K; ++k) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }

  return C;
}

bool frolova_e_matrix_multiplication_mpi::matrixMultiplicationSequential::pre_processing() {
  internal_order_test();

  // Init value for input and output
  int* value_1 = reinterpret_cast<int*>(taskData->inputs[0]);
  lineA = static_cast<size_t>(value_1[0]);
  columnA = static_cast<size_t>(value_1[1]);

  int* value_2 = reinterpret_cast<int*>(taskData->inputs[1]);
  lineB = static_cast<size_t>(value_2[0]);
  columnB = static_cast<size_t>(value_2[1]);

  int* matr1_ptr = reinterpret_cast<int*>(taskData->inputs[2]);
  matrixA.assign(matr1_ptr, matr1_ptr + taskData->inputs_count[2]);

  int* matr2_ptr = reinterpret_cast<int*>(taskData->inputs[3]);
  matrixB.assign(matr2_ptr, matr2_ptr + taskData->inputs_count[3]);

  matrixC.resize(lineA * columnB);

  return true;
}

bool frolova_e_matrix_multiplication_mpi::matrixMultiplicationSequential::validation() {
  internal_order_test();

  int* value_1 = reinterpret_cast<int*>(taskData->inputs[0]);
  if (taskData->inputs_count[0] != 2) {
    return false;
  }
  auto line1 = static_cast<size_t>(value_1[0]);
  auto column1 = static_cast<size_t>(value_1[1]);

  int* value_2 = reinterpret_cast<int*>(taskData->inputs[1]);
  if (taskData->inputs_count[1] != 2) {
    return false;
  }
  auto line2 = static_cast<size_t>(value_2[0]);
  auto column2 = static_cast<size_t>(value_2[1]);

  if (value_1[1] != value_2[0]) {
    return false;
  }
  if (taskData->inputs_count[2] != line1 * column1) {
    return false;
  }
  if (taskData->inputs_count[3] != line2 * column2) {
    return false;
  }
  if (taskData->outputs_count[0] != line1 * column2) {
    return false;
  }

  return true;
}

bool frolova_e_matrix_multiplication_mpi::matrixMultiplicationSequential::run() {
  internal_order_test();
  matrixC = Multiplication(lineA, columnB, columnA, matrixA, matrixB);

  return true;
}

bool frolova_e_matrix_multiplication_mpi::matrixMultiplicationSequential::post_processing() {
  internal_order_test();

  for (size_t i = 0; i < lineA * columnB; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = matrixC[i];
  }

  return true;
}

bool frolova_e_matrix_multiplication_mpi::matrixMultiplicationParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    int* value_1 = reinterpret_cast<int*>(taskData->inputs[0]);
    lineA = static_cast<size_t>(value_1[0]);
    columnA = static_cast<size_t>(value_1[1]);

    int* value_2 = reinterpret_cast<int*>(taskData->inputs[1]);
    lineB = static_cast<size_t>(value_2[0]);
    columnB = static_cast<size_t>(value_2[1]);

    int* matr1_ptr = reinterpret_cast<int*>(taskData->inputs[2]);
    matrixA.assign(matr1_ptr, matr1_ptr + taskData->inputs_count[2]);

    int* matr2_ptr = reinterpret_cast<int*>(taskData->inputs[3]);
    matrixB.assign(matr2_ptr, matr2_ptr + taskData->inputs_count[3]);

    matrixC.resize(lineA * columnB);
  }

  return true;
}

bool frolova_e_matrix_multiplication_mpi::matrixMultiplicationParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    int* value_1 = reinterpret_cast<int*>(taskData->inputs[0]);
    if (taskData->inputs_count[0] != 2) {
      return false;
    }
    auto line1 = static_cast<size_t>(value_1[0]);
    auto column1 = static_cast<size_t>(value_1[1]);

    int* value_2 = reinterpret_cast<int*>(taskData->inputs[1]);
    if (taskData->inputs_count[1] != 2) {
      return false;
    }
    auto line2 = static_cast<size_t>(value_2[0]);
    auto column2 = static_cast<size_t>(value_2[1]);

    if (value_1[1] != value_2[0]) {
      return false;
    }
    if (taskData->inputs_count[2] != line1 * column1) {
      return false;
    }
    if (taskData->inputs_count[3] != line2 * column2) {
      return false;
    }
    if (taskData->outputs_count[0] != line1 * column2) {
      return false;
    }
  }
  return true;
}

void frolova_e_matrix_multiplication_mpi::multiplyAndPlace(lineStruc& line, const columnStruc& column) {
  if (line.res_lines.size() != line.numberOfLines * line.outgoingLineLength) {
    line.res_lines.resize(line.numberOfLines * line.outgoingLineLength, 0);
  }
  for (size_t i = 0; i < line.numberOfLines; ++i) {
    for (size_t j = 0; j < column.numberOfColumns; ++j) {
      int sum = 0;

      for (size_t k = 0; k < line.enterLineslenght; ++k) {
        size_t lineIndex = i * line.enterLineslenght + k;
        size_t columnIndex = j * line.enterLineslenght + k;

        sum += line.local_lines[lineIndex] * column.local_columns[columnIndex];
      }

      size_t globalCol = column.index_colums[j];
      size_t localPos = i * line.outgoingLineLength + globalCol;

      line.res_lines[localPos] = sum;
    }
  }
}

bool frolova_e_matrix_multiplication_mpi::matrixMultiplicationParallel::run() {
  internal_order_test();

  broadcast(world, lineA, 0);
  broadcast(world, columnA, 0);
  broadcast(world, lineB, 0);
  broadcast(world, columnB, 0);

  unsigned int active_processes = 0;

  if (world.rank() == 0) {
    active_processes = std::min(world.size(), static_cast<int>(std::max(lineA, columnB)));
  }
  broadcast(world, active_processes, 0);

  unsigned int base_lines = 0;
  unsigned int remainder_lines = 0;
  unsigned int base_columns = 0;
  unsigned int remainder_columns = 0;

  if (world.rank() == 0) {
    base_lines = lineA / active_processes;
    remainder_lines = lineA % active_processes;
    base_columns = columnB / active_processes;
    remainder_columns = columnB % active_processes;
  }

  if (world.rank() == 0) {
    std::vector<int> lines_per_process(active_processes);
    std::vector<int> columns_per_process(active_processes);

    for (size_t i = 0; i < active_processes; ++i) {
      lines_per_process[i] = (i < remainder_lines) ? base_lines + 1 : base_lines;
      columns_per_process[i] = (i < remainder_columns) ? base_columns + 1 : base_columns;
    }

    std::vector<int> start_line_index(active_processes, 0);
    std::vector<int> start_column_index(active_processes, 0);

    for (size_t i = 1; i < active_processes; ++i) {
      start_line_index[i] = start_line_index[i - 1] + lines_per_process[i - 1];
      start_column_index[i] = start_column_index[i - 1] + columns_per_process[i - 1];
    }

    // Zero process
    localLinesA.numberOfLines = lines_per_process[0];
    localLinesA.enterLineslenght = columnA;
    localLinesA.outgoingLineLength = columnB;

    if (localLinesA.numberOfLines > 0) {
      int start_line = start_line_index[0];
      localLinesA.local_lines.assign(matrixA.begin() + start_line * columnA,
                                     matrixA.begin() + (start_line + localLinesA.numberOfLines) * columnA);
      localLinesA.index_lines.resize(localLinesA.numberOfLines);
      std::iota(localLinesA.index_lines.begin(), localLinesA.index_lines.end(), start_line);
      localLinesA.res_lines.resize(localLinesA.numberOfLines * columnB, 0);
    }

    localColumnB.numberOfColumns = columns_per_process[0];
    localColumnB.enterColumnLenght = lineB;

    if (localColumnB.numberOfColumns > 0) {
      int start_column = start_column_index[0];
      localColumnB.local_columns.resize(localColumnB.numberOfColumns * lineB);
      for (size_t col = 0; col < localColumnB.numberOfColumns; ++col) {
        for (size_t row = 0; row < lineB; ++row) {
          localColumnB.local_columns[col * lineB + row] = matrixB[row * columnB + start_column + col];
        }
      }
      localColumnB.index_colums.resize(localColumnB.numberOfColumns);
      std::iota(localColumnB.index_colums.begin(), localColumnB.index_colums.end(), start_column);
    }

    // Other processes
    for (unsigned int i = 1; i < active_processes; ++i) {
      lineStruc line_data;
      line_data.numberOfLines = lines_per_process[i];
      line_data.enterLineslenght = columnA;
      line_data.outgoingLineLength = columnB;

      if (line_data.numberOfLines > 0) {
        int start_line = start_line_index[i];
        line_data.local_lines.assign(matrixA.begin() + start_line * columnA,
                                     matrixA.begin() + (start_line + line_data.numberOfLines) * columnA);
        line_data.index_lines.resize(line_data.numberOfLines);
        std::iota(line_data.index_lines.begin(), line_data.index_lines.end(), start_line);
        line_data.res_lines.resize(line_data.numberOfLines * columnB);
      }

      world.send(i, 0, line_data);

      columnStruc column_data;
      column_data.numberOfColumns = columns_per_process[i];
      column_data.enterColumnLenght = lineB;

      if (column_data.numberOfColumns > 0) {
        int start_column = start_column_index[i];
        column_data.local_columns.resize(column_data.numberOfColumns * lineB);
        for (size_t col = 0; col < column_data.numberOfColumns; ++col) {
          for (size_t row = 0; row < lineB; ++row) {
            column_data.local_columns[col * lineB + row] = matrixB[row * columnB + start_column + col];
          }
        }
        column_data.index_colums.resize(column_data.numberOfColumns);
        std::iota(column_data.index_colums.begin(), column_data.index_colums.end(), start_column);
      }

      world.send(i, 0, column_data);
    }
  }

  if (world.rank() > 0 && world.rank() < static_cast<int>(active_processes)) {
    world.recv(0, 0, localLinesA);
    world.recv(0, 0, localColumnB);
  }

  // Matrix multiplication
  if (world.rank() < static_cast<int>(active_processes) && (active_processes > 1)) {
    int next_process = (world.rank() + 1) % active_processes;
    int prev_process = (world.rank() == 0) ? active_processes - 1 : world.rank() - 1;

    for (int step = 0; step < static_cast<int>(active_processes); ++step) {
      if (localLinesA.numberOfLines > 0 && localColumnB.numberOfColumns > 0) {
        multiplyAndPlace(localLinesA, localColumnB);
      }

      auto send_req = world.isend(next_process, 0, localColumnB);
      columnStruc col;
      world.recv(prev_process, 0, col);
      send_req.wait();
      localColumnB = col;
    }
  } else if (active_processes == 1) {
    multiplyAndPlace(localLinesA, localColumnB);
  }

  if (world.rank() > 0 && world.rank() < static_cast<int>(active_processes)) {
    world.send(0, 0, localLinesA);
  }

  if (world.rank() == 0) {
    std::vector<int> res_vec;
    res_vec.insert(res_vec.end(), localLinesA.res_lines.begin(), localLinesA.res_lines.end());

    for (unsigned int i = 1; i < active_processes; ++i) {
      lineStruc a;
      world.recv(i, 0, a);
      if (a.numberOfLines > 0) {
        res_vec.insert(res_vec.end(), a.res_lines.begin(), a.res_lines.end());
      }
    }
    matrixC = res_vec;
  }

  return true;
}

bool frolova_e_matrix_multiplication_mpi::matrixMultiplicationParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    for (size_t i = 0; i < lineA * columnB; i++) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = matrixC[i];
    }
  }

  return true;
}