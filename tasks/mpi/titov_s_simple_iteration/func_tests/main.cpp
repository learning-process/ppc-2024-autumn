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

  std::vector<std::vector<float>> global_matrix = {{10.0f, 5.0f, 20.0f}, {6.0f, -2.0f, 2.0f}};
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
    ASSERT_FALSE(seqTask.pre_processing());
  }
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

TEST(titov_s_simple_iteration_mpi, Test_Simple_Iteration_Parallel_3_3) {
  boost::mpi::communicator world;

  size_t matrix_size = 3;
  std::vector<double> global_result(matrix_size, 0.0);
  std::vector<double> Matrix = {10.0, 2.0, 3.0, 4.0, 20.0, 5.0, 1.0, 6.0, 15.0};
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
  ASSERT_TRUE(taskPar.validation());
  taskPar.pre_processing();
  taskPar.run();
  taskPar.post_processing();

  std::vector<std::vector<float>> global_matrix;
  global_matrix = {{10.0f, 2.0f, 3.0f, 3.0f}, {4.0f, 20.0f, 5.0f, 5.0f}, {1.0f, 6.0f, 15.0f, 4.0f}};
  float eps = 0.001f;
  size_t matrix_size_seq = 3;
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

TEST(titov_s_simple_iteration_mpi, Test_Simple_Iteration_Parallel_4_4) {
  boost::mpi::communicator world;

  size_t matrix_size = 4;
  std::vector<double> global_result(matrix_size, 0.0);
  std::vector<double> Matrix = {20.9, 1.2, 2.1, 0.9, 1.2, 21.2, 1.5, 2.5, 2.1, 1.5, 19.8, 1.3, 0.9, 2.5, 1.3, 32.1};
  std::vector<double> Values = {21.7, 27.46, 28.76, 49.72};
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
  global_matrix = {{20.9f, 1.2f, 2.1f, 0.9f, 21.7f},
                   {1.2f, 21.2f, 1.5f, 2.5f, 27.46f},
                   {2.1f, 1.5f, 19.8f, 1.3f, 28.76f},
                   {0.9f, 2.5f, 1.3f, 32.1f, 49.72f}};
  float eps = 0.001f;
  size_t matrix_size_seq = 4;
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

TEST(titov_s_simple_iteration_mpi, Test_Simple_Iteration_Parallel_5_5) {
  boost::mpi::communicator world;

  size_t matrix_size = 5;
  std::vector<double> global_result(matrix_size, 0.0);
  std::vector<double> Matrix = {20.0, 1.0, 2.0, 0.5, 1.5, 1.0,  25.0, 1.2, 0.8, 1.1, 2.0, 1.2, 30.0,
                                2.5,  0.9, 0.5, 0.8, 2.5, 22.0, 1.3,  1.5, 1.1, 0.9, 1.3, 26.0};
  std::vector<double> Values = {10.0, 15.0, 20.0, 18.0, 25.0};
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
  global_matrix = {{20.0f, 1.0f, 2.0f, 0.5f, 1.5f, 10.0f},
                   {1.0f, 25.0f, 1.2f, 0.8f, 1.1f, 15.0f},
                   {2.0f, 1.2f, 30.0f, 2.5f, 0.9f, 20.0f},
                   {0.5f, 0.8f, 2.5f, 22.0f, 1.3f, 18.0f},
                   {1.5f, 1.1f, 0.9f, 1.3f, 26.0f, 25.0f}};
  float eps = 0.001f;
  size_t matrix_size_seq = 5;
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

TEST(titov_s_simple_iteration_mpi, Test_Simple_Iteration_Parallel_10_10) {
  boost::mpi::communicator world;

  size_t matrix_size = 10;
  std::vector<double> global_result(matrix_size, 0.0);
  std::vector<double> Matrix = {
      25.0, 1.2, 2.1,  0.9, 1.3,  2.2, 1.1,  0.8, 1.0,  0.5, 1.2, 26.0, 1.5, 1.8,  0.7, 2.4,  1.3, 1.2,  1.0, 0.6,
      2.1,  1.5, 27.0, 2.0, 1.9,  1.2, 0.5,  1.1, 0.7,  0.8, 0.9, 1.8,  2.0, 30.0, 2.3, 1.9,  0.6, 0.8,  1.2, 1.1,
      1.3,  0.7, 1.9,  2.3, 28.0, 2.5, 1.4,  0.9, 0.6,  1.2, 2.2, 2.4,  1.2, 1.9,  2.5, 31.0, 0.7, 1.0,  1.1, 0.8,
      1.1,  1.3, 0.5,  0.6, 1.4,  0.7, 32.0, 1.2, 0.8,  1.3, 0.8, 1.2,  1.1, 0.8,  0.9, 1.0,  1.2, 29.0, 2.2, 1.1,
      1.0,  1.0, 0.7,  1.2, 0.6,  1.1, 0.8,  2.2, 33.0, 2.5, 0.5, 0.6,  0.8, 1.1,  1.2, 0.8,  1.3, 1.1,  2.5, 34.0};
  std::vector<double> Values = {10.0, 20.0, 15.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0};

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

  std::vector<std::vector<float>> global_matrix = {
      {25.0f, 1.2f, 2.1f, 0.9f, 1.3f, 2.2f, 1.1f, 0.8f, 1.0f, 0.5f, 10.0f},
      {1.2f, 26.0f, 1.5f, 1.8f, 0.7f, 2.4f, 1.3f, 1.2f, 1.0f, 0.6f, 20.0f},
      {2.1f, 1.5f, 27.0f, 2.0f, 1.9f, 1.2f, 0.5f, 1.1f, 0.7f, 0.8f, 15.0f},
      {0.9f, 1.8f, 2.0f, 30.0f, 2.3f, 1.9f, 0.6f, 0.8f, 1.2f, 1.1f, 25.0f},
      {1.3f, 0.7f, 1.9f, 2.3f, 28.0f, 2.5f, 1.4f, 0.9f, 0.6f, 1.2f, 30.0f},
      {2.2f, 2.4f, 1.2f, 1.9f, 2.5f, 31.0f, 0.7f, 1.0f, 1.1f, 0.8f, 35.0f},
      {1.1f, 1.3f, 0.5f, 0.6f, 1.4f, 0.7f, 32.0f, 1.2f, 0.8f, 1.3f, 40.0f},
      {0.8f, 1.2f, 1.1f, 0.8f, 0.9f, 1.0f, 1.2f, 29.0f, 2.2f, 1.1f, 45.0f},
      {1.0f, 1.0f, 0.7f, 1.2f, 0.6f, 1.1f, 0.8f, 2.2f, 33.0f, 2.5f, 50.0f},
      {0.5f, 0.6f, 0.8f, 1.1f, 1.2f, 0.8f, 1.3f, 1.1f, 2.5f, 34.0f, 55.0f}};

  float eps = 0.001f;
  size_t matrix_size_seq = 10;
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

TEST(titov_s_simple_iteration_mpi, Test_Simple_Iteration_Parallel_15_15) {
  boost::mpi::communicator world;

  size_t matrix_size = 15;
  std::vector<double> global_result(matrix_size, 0.0);
  std::vector<double> Matrix = {
      45.0, 1.2,  0.8,  1.5, 0.7,  0.9,  1.4, 1.1,  1.6,  1.3, 0.6,  1.0,  1.7, 1.2,  0.5,  1.1, 50.0, 1.3,  0.9,
      1.6,  0.7,  1.8,  0.8, 0.6,  1.0,  1.5, 1.9,  1.2,  0.8, 1.4,  0.9,  1.3, 52.0, 1.7,  1.8, 0.6,  1.1,  0.5,
      0.8,  1.0,  1.3,  1.5, 1.6,  1.2,  1.9, 1.0,  1.2,  1.5, 55.0, 0.9,  1.1, 1.8,  1.0,  1.4, 1.3,  0.7,  0.8,
      1.6,  1.5,  1.4,  1.2, 0.6,  1.7,  0.9, 53.0, 1.4,  1.5, 0.8,  1.1,  1.0, 1.9,  1.3,  0.9, 1.2,  1.8,  1.4,
      1.1,  0.8,  1.5,  1.4, 56.0, 1.6,  1.2, 0.5,  1.7,  0.9, 1.1,  1.0,  1.4, 1.3,  1.3,  1.8, 0.6,  1.0,  1.5,
      1.6,  54.0, 1.5,  1.2, 0.7,  1.3,  1.1, 0.8,  1.9,  1.4, 0.8,  0.6,  0.5, 1.8,  1.2,  1.1, 1.5,  51.0, 1.3,
      1.4,  1.9,  0.9,  1.5, 0.7,  1.1,  1.0, 1.3,  0.8,  1.4, 1.1,  0.5,  1.2, 1.3,  57.0, 0.8, 1.1,  1.4,  0.6,
      1.2,  1.3,  0.9,  1.2, 1.0,  1.3,  0.8, 1.7,  0.7,  1.4, 0.8,  58.0, 1.2, 1.1,  1.3,  1.5, 0.6,  1.6,  0.5,
      1.3,  1.1,  1.9,  1.4, 1.3,  1.9,  1.1, 1.2,  60.0, 0.9, 0.8,  1.3,  1.4, 1.1,  1.5,  1.6, 1.2,  0.6,  0.9,
      1.8,  1.1,  1.4,  1.3, 0.9,  62.0, 1.3, 1.0,  1.7,  0.9, 1.1,  1.4,  0.7, 1.2,  1.3,  0.8, 1.5,  0.9,  1.3,
      1.1,  1.3,  65.0, 1.5, 1.9,  1.4,  0.8, 1.3,  1.5,  1.1, 0.9,  0.7,  1.2, 1.3,  1.5,  0.9, 1.4,  1.6,  66.0,
      1.8,  1.5,  1.3,  0.9, 1.1,  0.8,  1.3, 1.2,  1.9,  1.4, 1.5,  1.6,  1.1, 1.8,  1.9,  68.0};
  std::vector<double> Values = {5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0};

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

  std::vector<std::vector<float>> global_matrix = {
      {45.0f, 1.2f, 0.8f, 1.5f, 0.7f, 0.9f, 1.4f, 1.1f, 1.6f, 1.3f, 0.6f, 1.0f, 1.7f, 1.2f, 0.5f, 5.0f},
      {1.1f, 50.0f, 1.3f, 0.9f, 1.6f, 0.7f, 1.8f, 0.8f, 0.6f, 1.0f, 1.5f, 1.9f, 1.2f, 0.8f, 1.4f, 6.0f},
      {0.9f, 1.3f, 52.0f, 1.7f, 1.8f, 0.6f, 1.1f, 0.5f, 0.8f, 1.0f, 1.3f, 1.5f, 1.6f, 1.2f, 1.9f, 7.0f},
      {1.0f, 1.2f, 1.5f, 55.0f, 0.9f, 1.1f, 1.8f, 1.0f, 1.4f, 1.3f, 0.7f, 0.8f, 1.6f, 1.5f, 1.4f, 8.0f},
      {1.2f, 0.6f, 1.7f, 0.9f, 53.0f, 1.4f, 1.5f, 0.8f, 1.1f, 1.0f, 1.9f, 1.3f, 0.9f, 1.2f, 1.8f, 9.0f},
      {1.4f, 1.1f, 0.8f, 1.5f, 1.4f, 56.0f, 1.6f, 1.2f, 0.5f, 1.7f, 0.9f, 1.1f, 1.0f, 1.4f, 1.3f, 10.0f},
      {1.3f, 1.8f, 0.6f, 1.0f, 1.5f, 1.6f, 54.0f, 1.5f, 1.2f, 0.7f, 1.3f, 1.1f, 0.8f, 1.9f, 1.4f, 11.0f},
      {0.8f, 0.6f, 0.5f, 1.8f, 1.2f, 1.1f, 1.5f, 51.0f, 1.3f, 1.4f, 1.9f, 0.9f, 1.5f, 0.7f, 1.1f, 12.0f},
      {1.0f, 1.3f, 0.8f, 1.4f, 1.1f, 0.5f, 1.2f, 1.3f, 57.0f, 0.8f, 1.1f, 1.4f, 0.6f, 1.2f, 1.3f, 13.0f},
      {0.9f, 1.2f, 1.0f, 1.3f, 0.8f, 1.7f, 0.7f, 1.4f, 0.8f, 58.0f, 1.2f, 1.1f, 1.3f, 1.5f, 0.6f, 14.0f},
      {1.6f, 0.5f, 1.3f, 1.1f, 1.9f, 1.4f, 1.3f, 1.9f, 1.1f, 1.2f, 60.0f, 0.9f, 0.8f, 1.3f, 1.4f, 15.0f},
      {1.1f, 1.5f, 1.6f, 1.2f, 0.6f, 0.9f, 1.8f, 1.1f, 1.4f, 1.3f, 0.9f, 62.0f, 1.3f, 1.0f, 1.7f, 16.0f},
      {0.9f, 1.1f, 1.4f, 0.7f, 1.2f, 1.3f, 0.8f, 1.5f, 0.9f, 1.3f, 1.1f, 1.3f, 65.0f, 1.5f, 1.9f, 17.0f},
      {1.4f, 0.8f, 1.3f, 1.5f, 1.1f, 0.9f, 0.7f, 1.2f, 1.3f, 1.5f, 0.9f, 1.4f, 1.6f, 66.0f, 1.8f, 18.0f},
      {1.5f, 1.3f, 0.9f, 1.1f, 0.8f, 1.3f, 1.2f, 1.9f, 1.4f, 1.5f, 1.6f, 1.1f, 1.8f, 1.9f, 68.0f, 19.0f}};

  float eps = 0.001f;
  size_t matrix_size_seq = 15;
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
