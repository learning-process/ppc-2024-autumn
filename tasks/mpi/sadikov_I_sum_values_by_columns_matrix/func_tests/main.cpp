#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <iostream>
#include <vector>

#include "mpi/sadikov_I_sum_values_by_columns_matrix/include/ops_mpi.h"

TEST(ParallelOperations, checkvalidation) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData;
  if (world.rank() == 0) {
    std::vector<int> in(900, 1);
    std::vector<int> in_index{30, 30};
    std::vector<int> out(30, 0);
    taskData = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out);
  }
  sadikov_I_Sum_values_by_columns_matrix_mpi::MPITask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
}

TEST(ParallelOperations, checkvalidation2) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData;
  if (world.rank() == 0) {
    std::vector<int> in(900, 1);
    std::vector<int> in_index{30, 30};
    std::vector<int> out(30, 0);
    taskData = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out);
  }
  sadikov_I_Sum_values_by_columns_matrix_mpi::MPITaskParallel sv(taskData);
  ASSERT_EQ(sv.validation(), true);
}

TEST(ParallelOperations, checkvalidation3) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData;
  if (world.rank() == 0) {
    std::vector<int> in(900, 1);
    std::vector<int> in_index{30, 30};
    std::vector<int> out(30, 0);
    taskData = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out);
  }
  sadikov_I_Sum_values_by_columns_matrix_mpi::MPITaskParallel sv(taskData);
  sadikov_I_Sum_values_by_columns_matrix_mpi::MPITask sv2(taskData);
  ASSERT_EQ(sv.validation(), true);
  ASSERT_EQ(sv2.validation(), true);
}

TEST(ParallelOperations, check_square_matrix) {
  boost::mpi::communicator world;
  const int columns = 130;
  const int rows = 130;
  std::vector<int> out_seq(columns, 0);
  std::vector<int> in(columns * rows, 1);
  std::vector<int> in_index{rows, columns};
  std::vector<int> out_par(columns, 0);
  auto taskData_seq = std::make_shared<ppc::core::TaskData>();
  auto taskData_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData_seq = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_seq);
    sadikov_I_Sum_values_by_columns_matrix_mpi::MPITask sv_seq(taskData_seq);
    ASSERT_EQ(sv_seq.validation(), true);
    sv_seq.pre_processing();
    sv_seq.run();
    sv_seq.post_processing();
    taskData_par = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_par);
  }
  //sadikov_I_Sum_values_by_columns_matrix_mpi::MPITaskParallel sv_par(taskData_par);
  //ASSERT_EQ(sv_par.validation(), true);
  //sv_par.pre_processing();
  //sv_par.run();
  //sv_par.post_processing();
  //if (world.rank() == 0) {
  //  ASSERT_EQ(out_seq, out_par);
  //}
}

TEST(ParallelOperations, check_rect_matrix) {
  boost::mpi::communicator world;
  const int columns = 80;
  const int rows = 50;
  std::vector<int> out_seq(columns, 0);
  std::vector<int> in;
  std::vector<int> in_index{rows, columns};
  std::vector<int> out_par(columns, 0);
  auto taskData_seq = std::make_shared<ppc::core::TaskData>();
  auto taskData_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = std::vector<int>(rows * columns, 1);
    taskData_par = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_par);
  }
  sadikov_I_Sum_values_by_columns_matrix_mpi::MPITaskParallel sv_par(taskData_par);
  ASSERT_EQ(sv_par.validation(), true);
  sv_par.pre_processing();
  sv_par.run();
  sv_par.post_processing();
  if (world.rank() == 0) {
    taskData_seq = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_seq);
    sadikov_I_Sum_values_by_columns_matrix_mpi::MPITask sv_seq(taskData_seq);
    ASSERT_EQ(sv_seq.validation(), true);
    sv_seq.pre_processing();
    sv_seq.run();
    sv_seq.post_processing();
    ASSERT_EQ(out_seq, out_par);
  }
}

TEST(ParallelOperations, check_square_matrix2) {
  boost::mpi::communicator world;
  const int columns = 50;
  const int rows = 50;
  std::vector<int> out_seq(columns, 0);
  std::vector<int> in = sadikov_I_Sum_values_by_columns_matrix_mpi::getRandomVector(columns * rows);
  std::vector<int> in_index{rows, columns};
  std::vector<int> out_par(columns, 0);
  auto taskData_seq = std::make_shared<ppc::core::TaskData>();
  auto taskData_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData_par = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_par);
  }
  sadikov_I_Sum_values_by_columns_matrix_mpi::MPITaskParallel sv_par(taskData_par);
  ASSERT_EQ(sv_par.validation(), true);
  sv_par.pre_processing();
  sv_par.run();
  sv_par.post_processing();
  if (world.rank() == 0) {
    taskData_seq = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_seq);
    sadikov_I_Sum_values_by_columns_matrix_mpi::MPITask sv_seq(taskData_seq);
    ASSERT_EQ(sv_seq.validation(), true);
    sv_seq.pre_processing();
    sv_seq.run();
    sv_seq.post_processing();
    ASSERT_EQ(out_seq, out_par);
  }
}

TEST(ParallelOperations, check_rect_matrix2) {
  boost::mpi::communicator world;
  const int columns = 50;
  const int rows = 260;
  std::vector<int> out_seq(columns, 0);
  std::vector<int> in = sadikov_I_Sum_values_by_columns_matrix_mpi::getRandomVector(columns * rows);
  std::vector<int> in_index{rows, columns};
  std::vector<int> out_par(columns, 0);
  auto taskData_seq = std::make_shared<ppc::core::TaskData>();
  auto taskData_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData_par = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_par);
  }
  sadikov_I_Sum_values_by_columns_matrix_mpi::MPITaskParallel sv_par(taskData_par);
  ASSERT_EQ(sv_par.validation(), true);
  sv_par.pre_processing();
  sv_par.run();
  sv_par.post_processing();
  if (world.rank() == 0) {
    taskData_seq = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_seq);
    sadikov_I_Sum_values_by_columns_matrix_mpi::MPITask sv_seq(taskData_seq);
    ASSERT_EQ(sv_seq.validation(), true);
    sv_seq.pre_processing();
    sv_seq.run();
    sv_seq.post_processing();
    ASSERT_EQ(out_seq, out_par);
  }
}

TEST(ParallelOperations, check_rand_matrix) {
  boost::mpi::communicator world;
  const int columns = 150;
  const int rows = 150;
  std::vector<int> out_seq(columns, 0);
  std::vector<int> in = sadikov_I_Sum_values_by_columns_matrix_mpi::getRandomVector(columns * rows);
  std::vector<int> in_index{rows, columns};
  std::vector<int> out_par(columns, 0);
  auto taskData_seq = std::make_shared<ppc::core::TaskData>();
  auto taskData_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData_par = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_par);
  }
  sadikov_I_Sum_values_by_columns_matrix_mpi::MPITaskParallel sv_par(taskData_par);
  ASSERT_EQ(sv_par.validation(), true);
  sv_par.pre_processing();
  sv_par.run();
  sv_par.post_processing();
  if (world.rank() == 0) {
    taskData_seq = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_seq);
    sadikov_I_Sum_values_by_columns_matrix_mpi::MPITask sv_seq(taskData_seq);
    ASSERT_EQ(sv_seq.validation(), true);
    sv_seq.pre_processing();
    sv_seq.run();
    sv_seq.post_processing();
    ASSERT_EQ(out_seq, out_par);
  }
}

TEST(ParallelOperations, check_rand_matrix2) {
  boost::mpi::communicator world;
  const int columns = 73;
  const int rows = 73;
  std::vector<int> out_seq(columns, 0);
  std::vector<int> in = sadikov_I_Sum_values_by_columns_matrix_mpi::getRandomVector(columns * rows);
  std::vector<int> in_index{rows, columns};
  std::vector<int> out_par(columns, 0);
  auto taskData_seq = std::make_shared<ppc::core::TaskData>();
  auto taskData_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData_par = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_par);
  }
  sadikov_I_Sum_values_by_columns_matrix_mpi::MPITaskParallel sv_par(taskData_par);
  ASSERT_EQ(sv_par.validation(), true);
  sv_par.pre_processing();
  sv_par.run();
  sv_par.post_processing();
  if (world.rank() == 0) {
    taskData_seq = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_seq);
    sadikov_I_Sum_values_by_columns_matrix_mpi::MPITask sv_seq(taskData_seq);
    ASSERT_EQ(sv_seq.validation(), true);
    sv_seq.pre_processing();
    sv_seq.run();
    sv_seq.post_processing();
    ASSERT_EQ(out_seq, out_par);
  }
}