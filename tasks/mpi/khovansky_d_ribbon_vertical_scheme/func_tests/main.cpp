// Copyright 2024 Khovansky Dmitry
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "mpi/khovansky_d_ribbon_vertical_scheme/include/ops_mpi.hpp"

TEST(khovansky_d_ribbon_vertical_scheme_mpi, procs_more_than_rows) {
  int rows_count = 3;
  int columns_count = 5;
  int proc_count = 6;

  std::vector<int> rows_per_process(proc_count, 0);
  std::vector<int> rows_offsets(proc_count, 0);

  if (proc_count > rows_count) {
    for (int i = 0; i < rows_count; ++i) {
      rows_offsets[i] = i * columns_count;
      rows_per_process[i] = columns_count;
    }
  } else {
    int rows_count_per_proc = rows_count / proc_count;
    int remainder = rows_count % proc_count;
    int offset = 0;
    for (int i = 0; i < proc_count; ++i) {
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

  std::vector<int> expected_rows_per_process = {5, 5, 5, 0, 0, 0};
  std::vector<int> expected_rows_offsets = {0, 5, 10, 0, 0, 0};

  EXPECT_EQ(rows_per_process, expected_rows_per_process);
  EXPECT_EQ(rows_offsets, expected_rows_offsets);
}

TEST(khovansky_d_ribbon_vertical_scheme_mpi, procs_less_than_rows) {
  int rows_count = 10;
  int columns_count = 3;
  int proc_count = 4;

  std::vector<int> rows_per_process(proc_count, 0);
  std::vector<int> rows_offsets(proc_count, 0);

  if (proc_count > rows_count) {
    for (int i = 0; i < rows_count; ++i) {
      rows_offsets[i] = i * columns_count;
      rows_per_process[i] = columns_count;
    }
  } else {
    int rows_count_per_proc = rows_count / proc_count;
    int remainder = rows_count % proc_count;
    int offset = 0;
    for (int i = 0; i < proc_count; ++i) {
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

  std::vector<int> expected_rows_per_process = {9, 9, 6, 6};
  std::vector<int> expected_rows_offsets = {0, 9, 18, 24};

  EXPECT_EQ(rows_per_process, expected_rows_per_process);
  EXPECT_EQ(rows_offsets, expected_rows_offsets);
}

TEST(khovansky_d_ribbon_vertical_scheme_mpi, procs_equal_rows) {
  int rows_count = 5;
  int columns_count = 3;
  int proc_count = 5;

  std::vector<int> rows_per_process(proc_count, 0);
  std::vector<int> rows_offsets(proc_count, 0);

  if (proc_count > rows_count) {
    for (int i = 0; i < rows_count; ++i) {
      rows_offsets[i] = i * columns_count;
      rows_per_process[i] = columns_count;
    }
  } else {
    int rows_count_per_proc = rows_count / proc_count;
    int remainder = rows_count % proc_count;
    int offset = 0;
    for (int i = 0; i < proc_count; ++i) {
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

  std::vector<int> expected_rows_per_process = {3, 3, 3, 3, 3};
  std::vector<int> expected_rows_offsets = {0, 3, 6, 9, 12};

  EXPECT_EQ(rows_per_process, expected_rows_per_process);
  EXPECT_EQ(rows_offsets, expected_rows_offsets);
}

TEST(khovansky_d_ribbon_vertical_scheme_mpi, standart_matrix) {
  boost::mpi::communicator world;

  int rows_count = 3;
  int columns_count = 3;
  std::vector<int> input_matrix;
  std::vector<int> input_vector;
  std::vector<int> output_vector;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_matrix.resize(rows_count * columns_count);
    input_vector.resize(rows_count);
    output_vector.resize(columns_count, 0);

    for (int i = 0; i < rows_count * columns_count; ++i) {
      input_matrix[i] = (rand() % 1000) - 500;
    }

    for (int i = 0; i < rows_count; ++i) {
      input_vector[i] = (rand() % 1000) - 500;
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
    taskDataPar->inputs_count.emplace_back(input_matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    taskDataPar->inputs_count.emplace_back(input_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
    taskDataPar->outputs_count.emplace_back(output_vector.size());
  }

  auto taskParallel = std::make_shared<khovansky_d_ribbon_vertical_scheme_mpi::RibbonVerticalSchemeMPI>(taskDataPar);

  ASSERT_TRUE(taskParallel->validation());
  ASSERT_TRUE(taskParallel->pre_processing());
  ASSERT_TRUE(taskParallel->run());
  ASSERT_TRUE(taskParallel->post_processing());
}