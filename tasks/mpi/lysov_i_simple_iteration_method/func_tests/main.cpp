// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/lysov_i_simple_iteration_method/include/ops_mpi.hpp"

TEST(lysov_i_simple_iteration_method_mpi, SlaeIterationTaskMPI_IterationConvergence) {
  boost::mpi::communicator world;
  const int input_size = 3;

  std::vector<double> matrix = {10.0, 1.0, -3.0, 2.0, 20.0, -1.0, -1.0, 1.0, 30.0};
  std::vector<double> g = {20.0, 30.0, 40.0};
  std::vector<double> x(input_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.push_back(input_size);
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(g.data()));
    taskDataPar->inputs_count.push_back(input_size);
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
    taskDataPar->outputs_count.push_back(input_size);
  }

  lysov_i_simple_iteration_method_mpi::SlaeIterationTaskMPI slaeIterationTaskMPI(taskDataPar);

  ASSERT_TRUE(slaeIterationTaskMPI.validation());
  slaeIterationTaskMPI.pre_processing();
  slaeIterationTaskMPI.run();
  slaeIterationTaskMPI.post_processing();

  const std::vector<double> expected_solution = {2.2752, 1.3406, 1.3644};

  if (world.rank() == 0) {
    for (int i = 0; i < input_size; ++i) {
      EXPECT_NEAR(x[i], expected_solution[i], 1e-4);
    }
  }
}

TEST(lysov_i_simple_iteration_method_mpi, test_empty_matrix) {
  boost::mpi::communicator world;
  const int input_size = 0;

  std::vector<double> matrix = {};
  std::vector<double> g = {};
  std::vector<double> x(input_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.push_back(input_size);
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(g.data()));
    taskDataPar->inputs_count.push_back(input_size);
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
    taskDataPar->outputs_count.push_back(input_size);
  }
  lysov_i_simple_iteration_method_mpi::SlaeIterationTaskMPI slaeIterationTaskMPI(taskDataPar);

  ASSERT_TRUE(slaeIterationTaskMPI.validation());
  slaeIterationTaskMPI.pre_processing();
  ASSERT_TRUE(slaeIterationTaskMPI.run());
  slaeIterationTaskMPI.post_processing();

  const std::vector<double> expected_solution = {};

  if (world.rank() == 0) {
    ASSERT_EQ(x, expected_solution);
  }
}

TEST(lysov_i_simple_iteration_method_mpi, test_zero_right) {
  boost::mpi::communicator world;
  const int input_size = 3;

  std::vector<double> matrix = {30.0, 2.0, 3.0, 5.0, 45.0, 2.0, 0.0, 10.0, 50.0};
  std::vector<double> g = {0.0, 0.0, 0.0};
  std::vector<double> x(input_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.push_back(input_size);
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(g.data()));
    taskDataPar->inputs_count.push_back(input_size);
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
    taskDataPar->outputs_count.push_back(input_size);
  }

  lysov_i_simple_iteration_method_mpi::SlaeIterationTaskMPI slaeIterationTaskMPI(taskDataPar);

  ASSERT_TRUE(slaeIterationTaskMPI.validation());
  slaeIterationTaskMPI.pre_processing();
  ASSERT_TRUE(slaeIterationTaskMPI.run());
  slaeIterationTaskMPI.post_processing();

  const std::vector<double> expected_solution = {0.0, 0.0, 0.0};

  if (world.rank() == 0) {
    for (int i = 0; i < input_size; ++i) {
      EXPECT_NEAR(x[i], expected_solution[i], 1e-6);
    }
  }
}

TEST(lysov_i_simple_iteration_method_mpi, test_negative_right) {
  boost::mpi::communicator world;
  const int input_size = 3;

  std::vector<double> matrix = {30.0, 2.0, 3.0, 5.0, 45.0, 2.0, 0.0, 10.0, 50.0};
  std::vector<double> g = {-1.0, -1.0, -1.0};
  std::vector<double> x(input_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.push_back(input_size);
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(g.data()));
    taskDataPar->inputs_count.push_back(input_size);
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
    taskDataPar->outputs_count.push_back(input_size);
  }

  lysov_i_simple_iteration_method_mpi::SlaeIterationTaskMPI slaeIterationTaskMPI(taskDataPar);

  ASSERT_TRUE(slaeIterationTaskMPI.validation());
  slaeIterationTaskMPI.pre_processing();
  ASSERT_TRUE(slaeIterationTaskMPI.run());
  slaeIterationTaskMPI.post_processing();

  const std::vector<double> expected_solution = {-0.03, -0.018, -0.016};

  if (world.rank() == 0) {
    for (int i = 0; i < input_size; ++i) {
      EXPECT_NEAR(x[i], expected_solution[i], 1e-3);
    }
  }
}