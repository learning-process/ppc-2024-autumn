#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <iostream>
#include <vector>

#include "mpi/sadikov_I_sum_values_by_columns_matrix/include/ops_mpi.h"

TEST(sadikov_I_Sum_values_by_columns_matrix_mpi, checkvalidation) {
  boost::mpi::communicator world;
  const int columns = 15;
  const int rows = 15;
  std::vector<int> in;
  std::vector<int> in_index;
  std::vector<int> out;
  if (world.rank() == 0) {
    in = std::vector<int>(rows * columns, 1);
    in_index = std::vector<int>{rows, columns};
    out = std::vector<int>(columns, 0);
  }
  auto taskData = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out);
  sadikov_I_Sum_values_by_columns_matrix_mpi::MPITask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
}

TEST(sadikov_I_Sum_values_by_columns_matrix_mpi, checkvalidation2) {
  boost::mpi::communicator world;
  const int columns = 15;
  const int rows = 15;
  std::vector<int> in;
  std::vector<int> in_index;
  std::vector<int> out;
  if (world.rank() == 0) {
    in = std::vector<int>(rows * columns, 1);
    in_index = std::vector<int>{rows, columns};
    out = std::vector<int>(columns, 0);
  }
  auto taskData = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out);
  sadikov_I_Sum_values_by_columns_matrix_mpi::MPITaskParallel sv(taskData);
  ASSERT_EQ(sv.validation(), true);
}

TEST(sadikov_I_Sum_values_by_columns_matrix_mpi, check_square_matrix) {
  boost::mpi::communicator world;
  const int columns = 15;
  const int rows = 15;
  std::vector<int> out_seq;
  std::vector<int> in;
  std::vector<int> in_index;
  std::vector<int> out_par;
  in = std::vector<int>(rows * columns, 1);
  in_index = std::vector<int>{rows, columns};
  out_seq = std::vector<int>(columns, 0);
  out_par = std::vector<int>(columns, 0);
  auto taskData_par = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_par);
  sadikov_I_Sum_values_by_columns_matrix_mpi::MPITaskParallel sv_par(taskData_par);
  ASSERT_EQ(sv_par.validation(), true);
  sv_par.pre_processing();
  sv_par.run();
  sv_par.post_processing();
  if (world.rank() == 0) {
    auto taskData_seq = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_seq);
    sadikov_I_Sum_values_by_columns_matrix_mpi::MPITask sv_seq(taskData_seq);
    ASSERT_EQ(sv_seq.validation(), true);
    sv_seq.pre_processing();
    sv_seq.run();
    sv_seq.post_processing();
    ASSERT_EQ(out_seq, out_par);
  }
}

TEST(sadikov_I_Sum_values_by_columns_matrix_mpi, check_rect_matrix) {
  boost::mpi::communicator world;
  const int columns = 15;
  const int rows = 40;
  std::vector<int> out_seq;
  std::vector<int> in;
  std::vector<int> in_index;
  std::vector<int> out_par;
  in = std::vector<int>(rows * columns, 1);
  in_index = std::vector<int>{rows, columns};
  out_seq = std::vector<int>(columns, 0);
  out_par = std::vector<int>(columns, 0);
  auto taskData_par = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_par);
  sadikov_I_Sum_values_by_columns_matrix_mpi::MPITaskParallel sv_par(taskData_par);
  ASSERT_EQ(sv_par.validation(), true);
  sv_par.pre_processing();
  sv_par.run();
  sv_par.post_processing();
  if (world.rank() == 0) {
    auto taskData_seq = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_seq);
    sadikov_I_Sum_values_by_columns_matrix_mpi::MPITask sv_seq(taskData_seq);
    ASSERT_EQ(sv_seq.validation(), true);
    sv_seq.pre_processing();
    sv_seq.run();
    sv_seq.post_processing();
    ASSERT_EQ(out_seq, out_par);
  }
}

TEST(sadikov_I_Sum_values_by_columns_matrix_mpi, check_rect_matrix2) {
  boost::mpi::communicator world;
  const int columns = 150;
  const int rows = 40;
  std::vector<int> out_seq;
  std::vector<int> in;
  std::vector<int> in_index;
  std::vector<int> out_par;
  in = sadikov_I_Sum_values_by_columns_matrix_mpi::getRandomVector(columns * rows);
  in_index = std::vector<int>{rows, columns};
  out_seq = std::vector<int>(columns, 0);
  out_par = std::vector<int>(columns, 0);
  auto taskData_par = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_par);
  sadikov_I_Sum_values_by_columns_matrix_mpi::MPITaskParallel sv_par(taskData_par);
  ASSERT_EQ(sv_par.validation(), true);
  sv_par.pre_processing();
  sv_par.run();
  sv_par.post_processing();
  if (world.rank() == 0) {
    auto taskData_seq = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_seq);
    sadikov_I_Sum_values_by_columns_matrix_mpi::MPITask sv_seq(taskData_seq);
    ASSERT_EQ(sv_seq.validation(), true);
    sv_seq.pre_processing();
    sv_seq.run();
    sv_seq.post_processing();
    ASSERT_EQ(out_seq, out_par);
  }
}

TEST(sadikov_I_Sum_values_by_columns_matrix_mpi, check_rect_matrix3) {
  boost::mpi::communicator world;
  const int columns = 179;
  const int rows = 91;
  std::vector<int> out_seq;
  std::vector<int> in;
  std::vector<int> in_index;
  std::vector<int> out_par;
  in = sadikov_I_Sum_values_by_columns_matrix_mpi::getRandomVector(columns * rows);
  in_index = std::vector<int>{rows, columns};
  out_seq = std::vector<int>(columns, 0);
  out_par = std::vector<int>(columns, 0);
  auto taskData_par = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_par);
  sadikov_I_Sum_values_by_columns_matrix_mpi::MPITaskParallel sv_par(taskData_par);
  ASSERT_EQ(sv_par.validation(), true);
  sv_par.pre_processing();
  sv_par.run();
  sv_par.post_processing();
  if (world.rank() == 0) {
    auto taskData_seq = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_seq);
    sadikov_I_Sum_values_by_columns_matrix_mpi::MPITask sv_seq(taskData_seq);
    ASSERT_EQ(sv_seq.validation(), true);
    sv_seq.pre_processing();
    sv_seq.run();
    sv_seq.post_processing();
    ASSERT_EQ(out_seq, out_par);
  }
}

TEST(sadikov_I_Sum_values_by_columns_matrix_mpi, check_square_matrix2) {
  boost::mpi::communicator world;
  const int columns = 153;
  const int rows = 153;
  std::vector<int> out_seq;
  std::vector<int> in;
  std::vector<int> in_index;
  std::vector<int> out_par;
  in = sadikov_I_Sum_values_by_columns_matrix_mpi::getRandomVector(columns * rows);
  in_index = std::vector<int>{rows, columns};
  out_seq = std::vector<int>(columns, 0);
  out_par = std::vector<int>(columns, 0);
  auto taskData_par = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_par);
  sadikov_I_Sum_values_by_columns_matrix_mpi::MPITaskParallel sv_par(taskData_par);
  ASSERT_EQ(sv_par.validation(), true);
  sv_par.pre_processing();
  sv_par.run();
  sv_par.post_processing();
  if (world.rank() == 0) {
    auto taskData_seq = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_seq);
    sadikov_I_Sum_values_by_columns_matrix_mpi::MPITask sv_seq(taskData_seq);
    ASSERT_EQ(sv_seq.validation(), true);
    sv_seq.pre_processing();
    sv_seq.run();
    sv_seq.post_processing();
    ASSERT_EQ(out_seq, out_par);
  }
}

TEST(sadikov_I_Sum_values_by_columns_matrix_mpi, check_square_matrix3) {
  boost::mpi::communicator world;
  const int columns = 450;
  const int rows = 450;
  std::vector<int> out_seq;
  std::vector<int> in;
  std::vector<int> in_index;
  std::vector<int> out_par;
  in = sadikov_I_Sum_values_by_columns_matrix_mpi::getRandomVector(columns * rows);
  in_index = std::vector<int>{rows, columns};
  out_seq = std::vector<int>(columns, 0);
  out_par = std::vector<int>(columns, 0);
  auto taskData_par = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_par);
  sadikov_I_Sum_values_by_columns_matrix_mpi::MPITaskParallel sv_par(taskData_par);
  ASSERT_EQ(sv_par.validation(), true);
  sv_par.pre_processing();
  sv_par.run();
  sv_par.post_processing();
  if (world.rank() == 0) {
    auto taskData_seq = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_seq);
    sadikov_I_Sum_values_by_columns_matrix_mpi::MPITask sv_seq(taskData_seq);
    ASSERT_EQ(sv_seq.validation(), true);
    sv_seq.pre_processing();
    sv_seq.run();
    sv_seq.post_processing();
    ASSERT_EQ(out_seq, out_par);
  }
}