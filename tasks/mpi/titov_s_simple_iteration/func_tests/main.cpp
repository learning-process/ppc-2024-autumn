// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/titov_s_simple_iteration/include/ops_mpi.hpp"

TEST(titov_s_simple_iteration_mpi, Test_Simple_Iteration_Not_A_Diagonally_Dominate) {
  boost::mpi::communicator world;
  size_t matrix_size = 3;
  std::vector<double> global_result(matrix_size, 0.0);
  std::vector<double> Matrix = {1.0, 2.0, 3.0, 14.0, 2.8, 35.0, 1.0, 6.0, 0.1};
  std::vector<double> Values = {3.0, 5.0, 4.0};
  double epsilon = 0.001;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(Matrix.data()));
    taskDataPar->inputs_count.emplace_back(Matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(Values.data()));
    taskDataPar->inputs_count.emplace_back(Values.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  titov_s_simple_iteration_mpi::MPISimpleIterationParallel taskPar(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_FALSE(taskPar.validation());
  }
}

TEST(titov_s_simple_iteration_mpi, Test_Simple_Iteration_Parallel_Empty) {
  boost::mpi::communicator world;

  size_t matrix_size = 0;
  std::vector<double> global_result(matrix_size, 0.0);
  std::vector<double> Matrix = {};
  std::vector<double> Values = {};
  double epsilon = 0.001;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(Matrix.data()));
    taskDataPar->inputs_count.emplace_back(Matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(Values.data()));
    taskDataPar->inputs_count.emplace_back(Values.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  titov_s_simple_iteration_mpi::MPISimpleIterationParallel taskPar(taskDataPar);
  ASSERT_FALSE(taskPar.validation());
}

TEST(titov_s_simple_iteration_mpi, Test_Simple_Iteration_Parallel_1_1) {
  boost::mpi::communicator world;

  size_t matrix_size = 1;
  std::vector<double> global_result(matrix_size, 0.0);
  std::vector<double> Matrix = {10.0};
  std::vector<double> Values = {1.0};
  double epsilon = 0.001;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(Matrix.data()));
    taskDataPar->inputs_count.emplace_back(Matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(Values.data()));
    taskDataPar->inputs_count.emplace_back(Values.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  titov_s_simple_iteration_mpi::MPISimpleIterationParallel taskPar(taskDataPar);
  ASSERT_TRUE(taskPar.validation());
  taskPar.pre_processing();
  taskPar.run();
  taskPar.post_processing();

  std::vector<std::vector<float>> global_matrix;
  global_matrix = {{10.0f, 1.0f}};
  float eps = 0.001f;
  size_t matrix_size_seq = 1;

  std::vector<float> expected_result(matrix_size_seq, 0.0f);
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    for (const auto& row : global_matrix) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<float*>(row.data())));
    }

    taskDataSeq->inputs_count.push_back(global_matrix.size());
    taskDataSeq->inputs_count.push_back(global_matrix[0].size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expected_result.data()));
    taskDataSeq->outputs_count.push_back(expected_result.size());

    titov_s_simple_iteration_mpi::MPISimpleIterationSequential seqTask(taskDataSeq);

    ASSERT_TRUE(seqTask.validation());
    seqTask.pre_processing();
    seqTask.run();
    seqTask.post_processing();
  }
  if (world.rank() == 0) {
    for (unsigned int i = 0; i < global_result.size(); ++i) {
      ASSERT_NEAR(global_result[i], expected_result[i], epsilon);
    }
  }
}

TEST(titov_s_simple_iteration_mpi, Test_Simple_Iteration_Parallel_2_2) {
  boost::mpi::communicator world;

  size_t matrix_size = 2;
  std::vector<double> global_result(matrix_size, 0.0);
  std::vector<double> Matrix = {10.0, 2.0, 3.0, 20.0};
  std::vector<double> Values = {3.0, 5.0};
  double epsilon = 0.001;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(Matrix.data()));
    taskDataPar->inputs_count.emplace_back(Matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(Values.data()));
    taskDataPar->inputs_count.emplace_back(Values.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  titov_s_simple_iteration_mpi::MPISimpleIterationParallel taskPar(taskDataPar);
  ASSERT_TRUE(taskPar.validation());
  taskPar.pre_processing();
  taskPar.run();
  taskPar.post_processing();

  std::vector<std::vector<float>> global_matrix;
  global_matrix = {{10.0f, 2.0f, 3.0f}, {3.0f, 20.0f, 5.0f}};
  float eps = 0.001f;
  size_t matrix_size_seq = 2;
  std::vector<float> expected_result(matrix_size_seq, 0.0f);
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    for (const auto& row : global_matrix) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<float*>(row.data())));
    }

    taskDataSeq->inputs_count.push_back(global_matrix.size());
    taskDataSeq->inputs_count.push_back(global_matrix[0].size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expected_result.data()));
    taskDataSeq->outputs_count.push_back(expected_result.size());

    titov_s_simple_iteration_mpi::MPISimpleIterationSequential seqTask(taskDataSeq);

    ASSERT_TRUE(seqTask.validation());
    seqTask.pre_processing();
    seqTask.run();
    seqTask.post_processing();
  }
  if (world.rank() == 0) {
    for (unsigned int i = 0; i < global_result.size(); ++i) {
      ASSERT_NEAR(global_result[i], expected_result[i], epsilon);
    }
  }
}
