// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/borisov_s_sum_of_rows/include/ops_mpi.hpp"

TEST(borisov_s_sum_of_rows, Test_Unit_Matrix) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_row_sums;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  size_t rows = 10;
  size_t cols = 10;

  if (world.rank() == 0) {
    global_matrix.resize(rows * cols, 1);
    global_row_sums.resize(rows, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_row_sums.data()));
    taskDataPar->outputs_count.push_back(global_row_sums.size());
  } else {
    taskDataPar->inputs.emplace_back(nullptr);
    taskDataPar->outputs.emplace_back(nullptr);
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs_count.push_back(0);
  }

  borisov_s_sum_of_rows::SumOfRowsTaskParallel sumOfRowsTaskParallel(taskDataPar);
  ASSERT_EQ(sumOfRowsTaskParallel.validation(), true);

  sumOfRowsTaskParallel.pre_processing();
  sumOfRowsTaskParallel.run();
  sumOfRowsTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (size_t i = 0; i < global_row_sums.size(); i++) {
      ASSERT_EQ(global_row_sums[i], cols);
    }
  }
}

TEST(borisov_s_sum_of_rows, Test_Zero_Matrix) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_row_sums;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  size_t rows = 10;
  size_t cols = 10;

  if (world.rank() == 0) {
    global_matrix.resize(rows * cols, 0);
    global_row_sums.resize(rows, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_row_sums.data()));
    taskDataPar->outputs_count.push_back(global_row_sums.size());
  } else {
    taskDataPar->inputs.emplace_back(nullptr);
    taskDataPar->outputs.emplace_back(nullptr);
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs_count.push_back(0);
  }

  borisov_s_sum_of_rows::SumOfRowsTaskParallel sumOfRowsTaskParallel(taskDataPar);
  ASSERT_EQ(sumOfRowsTaskParallel.validation(), true);

  sumOfRowsTaskParallel.pre_processing();
  sumOfRowsTaskParallel.run();
  sumOfRowsTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (size_t i = 0; i < global_row_sums.size(); i++) {
      ASSERT_EQ(global_row_sums[i], 0);
    }
  }
}

TEST(borisov_s_sum_of_rows, Test_Sum_Rows) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_row_sums;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  size_t rows = 15;
  size_t cols = 15;

  if (world.rank() == 0) {
    global_matrix = borisov_s_sum_of_rows::getRandomMatrix(rows, cols);
    global_row_sums.resize(rows, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_row_sums.data()));
    taskDataPar->outputs_count.push_back(global_row_sums.size());
  } else {
    taskDataPar->inputs.emplace_back(nullptr);
    taskDataPar->outputs.emplace_back(nullptr);
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs_count.push_back(0);
  }

  borisov_s_sum_of_rows::SumOfRowsTaskParallel sumOfRowsTaskParallel(taskDataPar);
  ASSERT_EQ(sumOfRowsTaskParallel.validation(), true);

  sumOfRowsTaskParallel.pre_processing();
  sumOfRowsTaskParallel.run();
  sumOfRowsTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_row_sums(global_row_sums.size(), 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.push_back(rows);
    taskDataSeq->inputs_count.push_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_row_sums.data()));
    taskDataSeq->outputs_count.push_back(reference_row_sums.size());

    borisov_s_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTaskSequential(taskDataSeq);
    ASSERT_EQ(sumOfRowsTaskSequential.validation(), true);
    sumOfRowsTaskSequential.pre_processing();
    sumOfRowsTaskSequential.run();
    sumOfRowsTaskSequential.post_processing();

    for (size_t i = 0; i < global_row_sums.size(); i++) {
      ASSERT_EQ(reference_row_sums[i], global_row_sums[i]);
    }
  }
}

TEST(borisov_s_sum_of_rows, Test_Empty_Matrix) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_row_sums;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    size_t rows = 0;
    size_t cols = 0;
    global_row_sums.resize(rows, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_row_sums.data()));
    taskDataPar->outputs_count.push_back(global_row_sums.size());
  }

  borisov_s_sum_of_rows::SumOfRowsTaskParallel sumOfRowsTaskParallel(taskDataPar);

  ASSERT_FALSE(sumOfRowsTaskParallel.validation());

  sumOfRowsTaskParallel.pre_processing();
  sumOfRowsTaskParallel.run();
  sumOfRowsTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_TRUE(global_row_sums.empty());
  }
}