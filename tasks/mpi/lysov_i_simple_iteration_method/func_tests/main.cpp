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

  if (world.rank() == 0) {
    std::vector<double> reference_result(input_size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.push_back(input_size);
    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(g.data()));
    taskDataSeq->inputs_count.push_back(input_size);
    taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.push_back(input_size);
    lysov_i_simple_iteration_method_mpi::SlaeIterationTask testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (int i = 0; i < input_size; ++i) {
      EXPECT_NEAR(x[i], reference_result[i], 1e-3);
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

  if (world.rank() == 0) {
    std::vector<double> reference_result(input_size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.push_back(input_size);
    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(g.data()));
    taskDataSeq->inputs_count.push_back(input_size);
    taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.push_back(input_size);
    lysov_i_simple_iteration_method_mpi::SlaeIterationTask testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (int i = 0; i < input_size; ++i) {
      EXPECT_NEAR(x[i], reference_result[i], 1e-3);
    }
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

  if (world.rank() == 0) {
    std::vector<double> reference_result(input_size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.push_back(input_size);
    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(g.data()));
    taskDataSeq->inputs_count.push_back(input_size);
    taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.push_back(input_size);
    lysov_i_simple_iteration_method_mpi::SlaeIterationTask testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (int i = 0; i < input_size; ++i) {
      EXPECT_NEAR(x[i], reference_result[i], 1e-3);
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
  if (world.rank() == 0) {
    std::vector<double> reference_result(input_size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.push_back(input_size);
    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(g.data()));
    taskDataSeq->inputs_count.push_back(input_size);
    taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.push_back(input_size);
    lysov_i_simple_iteration_method_mpi::SlaeIterationTask testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (int i = 0; i < input_size; ++i) {
      EXPECT_NEAR(x[i], reference_result[i], 1e-3);
    }
  }
}
