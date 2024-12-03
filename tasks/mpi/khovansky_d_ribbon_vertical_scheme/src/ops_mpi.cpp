// Copyright 2024 Khovansky Dmitry
#include "mpi/khovansky_d_ribbon_vertical_scheme/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <iostream>
#include <random>
#include <string>

bool khovansky_d_ribbon_vertical_scheme_mpi::RibbonVerticalSchemeSeq::validation() {
  internal_order_test();

  if (!taskData) {
    return false;
  }

  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 &&
         taskData->inputs_count[0] % taskData->inputs_count[1] == 0;
}

bool khovansky_d_ribbon_vertical_scheme_mpi::RibbonVerticalSchemeSeq::pre_processing() {
  internal_order_test();

  hello_matrix = reinterpret_cast<int*>(taskData->inputs[0]);
  hello_vector = reinterpret_cast<int*>(taskData->inputs[1]);

  int matrix_elements_count = taskData->inputs_count[0];
  rows_count = taskData->inputs_count[1];
  columns_count = matrix_elements_count / rows_count;

  goodbye_vector.assign(columns_count, 0);

  return true;
}

bool khovansky_d_ribbon_vertical_scheme_mpi::RibbonVerticalSchemeSeq::run() {
  internal_order_test();

  for (int i = 0; i < rows_count; ++i) {
    for (int j = 0; j < columns_count; ++j) {
      goodbye_vector[j] += hello_matrix[i * columns_count + j] * hello_vector[i];
    }
  }

  return true;
}

bool khovansky_d_ribbon_vertical_scheme_mpi::RibbonVerticalSchemeSeq::post_processing() {
  internal_order_test();

  int* result = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(goodbye_vector.begin(), goodbye_vector.end(), result);

  return true;
}

bool khovansky_d_ribbon_vertical_scheme_mpi::RibbonVerticalSchemeMPI::validation() {
  if (world.rank() != 0) return true;

  if (!taskData) {
    return false;
  }

  if (taskData->inputs[0] == nullptr && taskData->inputs_count[0] == 0) {
    return false;
  }
  if (taskData->inputs[1] == nullptr && taskData->inputs_count[1] == 0) {
    return false;
  }

  if (taskData->outputs[0] == nullptr) {
    return false;
  }

  return taskData->inputs_count[0] % taskData->inputs_count[1] == 0;
}

bool khovansky_d_ribbon_vertical_scheme_mpi::RibbonVerticalSchemeMPI::pre_processing() {
  if (world.rank() == 0) {
    if (!taskData || taskData->inputs[0] == nullptr || taskData->inputs[1] == nullptr ||
        taskData->outputs[0] == nullptr) {
      return false;
    }
    int* temp_matrix = reinterpret_cast<int*>(taskData->inputs[0]);
    int* temp_vector = reinterpret_cast<int*>(taskData->inputs[1]);

    int matrix_elements_count = taskData->inputs_count[0];
    rows_count = taskData->inputs_count[1];
    columns_count = matrix_elements_count / rows_count;

    hello_matrix.assign(temp_matrix, temp_matrix + matrix_elements_count);
    hello_vector.assign(temp_vector, temp_vector + rows_count);
    goodbye_vector.resize(columns_count, 0);

    rows_per_process.resize(world.size(), 0);
    rows_offsets.resize(world.size(), -1);

    if (world.size() > rows_count) {
      for (int i = 0; i < rows_count; ++i) {
        rows_offsets[i] = i * columns_count;
        rows_per_process[i] = columns_count;
      }
      for (int i = rows_count; i < world.size(); ++i) {
        rows_offsets[i] = -1;
        rows_per_process[i] = 0;
      }
    } else {
      int rows_count_per_proc = rows_count / world.size();
      int remainder = rows_count % world.size();
      int offset = 0;
      for (int i = 0; i < world.size(); ++i) {
        if (remainder > 0) {
          rows_per_process[i] = (rows_count_per_proc + 1) * columns_count;
          --remainder;
        } else {
          rows_per_process[i] = rows_count_per_proc * columns_count;
        }
        rows_offsets[i] = offset;
        offset += rows_per_process[i];
      }
    }
  }

  return true;
}

bool khovansky_d_ribbon_vertical_scheme_mpi::RibbonVerticalSchemeMPI::run() {
  boost::mpi::broadcast(world, rows_count, 0);
  boost::mpi::broadcast(world, columns_count, 0);
  boost::mpi::broadcast(world, rows_per_process, 0);
  boost::mpi::broadcast(world, rows_offsets, 0);
  boost::mpi::broadcast(world, hello_matrix, 0);
  boost::mpi::broadcast(world, hello_vector, 0);

  int process_start = rows_offsets[world.rank()] / columns_count;
  int matrix_start_point = rows_per_process[world.rank()] / columns_count;
  std::vector<int> process_result(columns_count, 0);

  for (int i = 0; i < columns_count; ++i) {
    for (int j = 0; j < matrix_start_point; ++j) {
      int prog_start = process_start + j;
      if (prog_start < rows_count) {
        int matrix = hello_matrix[i * rows_count + prog_start];
        int vector = hello_vector[prog_start];
        process_result[j] += matrix * vector;
      }
    }
  }

  boost::mpi::reduce(world, process_result.data(), columns_count, goodbye_vector.data(), std::plus<>(), 0);

  return true;
}

bool khovansky_d_ribbon_vertical_scheme_mpi::RibbonVerticalSchemeMPI::post_processing() {
  if (world.rank() == 0) {
    int* goodbye = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(goodbye_vector.begin(), goodbye_vector.end(), goodbye);
  }

  return true;
}