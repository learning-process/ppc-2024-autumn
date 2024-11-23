// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/malyshev_a_simple_iteration_method/include/ops_mpi.hpp"

TEST(malyshev_a_simple_iteration_method_mpi, basic_test) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  std::vector<double> A{3, 2, -1, 1, -2, 1, 2, -3, -5};
  std::vector<double> X(3, 0);
  std::vector<double> B{8, -2, 1};
  double eps = 1e-4;
  if (world.rank() == 0) {
    // Create data
    std::vector<double> X0(3, 0);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
    taskDataPar->inputs_count.emplace_back(X.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->outputs_count.emplace_back(X.size());
  }
  // Create Task
  malyshev_a_simple_iteration_method_mpi::TestMPITaskParallel testTaskSequential(taskDataPar);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  if (world.rank() == 0) {
    double sum_eq;
    for (uint32_t i = 0; i < X.size(); i++) {
      sum_eq = 0;
      for (uint32_t j = 0; j < X.size(); j++) {
        sum_eq += X[j] * A[i * X.size() + j];
      }
      ASSERT_TRUE(std::abs(sum_eq - B[i]) <= eps);
    }
  }
}

TEST(malyshev_a_simple_iteration_method_mpi, random_test) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> X;
  double eps = 1e-4;
  if (world.rank() == 0) {
    // Create data
    const int size = 10;
    X.resize(size);
    malyshev_a_simple_iteration_method_mpi::getRandomData(size, A, B);
    std::vector<double> X0(size, 0);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
    taskDataPar->inputs_count.emplace_back(X.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->outputs_count.emplace_back(X.size());
  }
  // Create Task
  malyshev_a_simple_iteration_method_mpi::TestMPITaskParallel testTaskSequential(taskDataPar);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  if (world.rank() == 0) {
    double sum_eq;
    for (uint32_t i = 0; i < X.size(); i++) {
      sum_eq = 0;
      for (uint32_t j = 0; j < X.size(); j++) {
        sum_eq += X[j] * A[i * X.size() + j];
      }
      ASSERT_TRUE(std::abs(sum_eq - B[i]) <= eps);
    }
  }
}