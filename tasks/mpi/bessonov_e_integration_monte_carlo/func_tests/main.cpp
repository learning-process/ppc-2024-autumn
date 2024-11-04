#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <random>

#include "mpi/bessonov_e_integration_monte_carlo/include/ops_mpi.hpp"

TEST(bessonov_e_integration_monte_carlo_mpi, PositiveRangeTestMPI) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double a = 0.0;
  double b = 1.0;
  int num_points = 1000000;
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_points));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }
  bessonov_e_integration_monte_carlo_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_points));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    bessonov_e_integration_monte_carlo_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_NEAR(reference_result[0], global_result[0], 1e-1);
  }
}

TEST(bessonov_e_integration_monte_carlo_mpi, NegativeRangeTestMPI) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double a = -1.0;
  double b = 0.0;
  int num_points = 100000;
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_points));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }
  bessonov_e_integration_monte_carlo_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_points));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    bessonov_e_integration_monte_carlo_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_NEAR(reference_result[0], global_result[0], 1e-1);
  }
}

TEST(bessonov_e_integration_monte_carlo_mpi, VerySmallRangeTestMPI) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double a = 0.1;
  double b = 0.11;
  int num_points = 100000;
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_points));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }
  bessonov_e_integration_monte_carlo_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_points));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    bessonov_e_integration_monte_carlo_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_NEAR(reference_result[0], global_result[0], 3e-8);
  }
}

TEST(bessonov_e_integration_monte_carlo_mpi, LongRangeTestMPI) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double a = -10.0;
  double b = 15.0;
  int num_points = 100000;
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_points));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }
  bessonov_e_integration_monte_carlo_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_points));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    bessonov_e_integration_monte_carlo_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_NEAR(reference_result[0], global_result[0], 1e3);
  }
}

TEST(bessonov_e_integration_monte_carlo_mpi, VeryLongRangeTestMPI) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double a = -40.0;
  double b = 50.0;
  int num_points = 1000000;
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_points));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }
  bessonov_e_integration_monte_carlo_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_points));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    bessonov_e_integration_monte_carlo_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_NEAR(reference_result[0], global_result[0], 3e4);
  }
}

TEST(bessonov_e_integration_monte_carlo_mpi, EqualRangeTestMPI) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double a = -2.0;
  double b = 2.0;
  int num_points = 100000;
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_points));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }
  bessonov_e_integration_monte_carlo_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_points));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    bessonov_e_integration_monte_carlo_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_NEAR(reference_result[0], global_result[0], 2e-1);
  }
}

TEST(bessonov_e_integration_monte_carlo_mpi, RandomTestMPI) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-8.0, 8.0);
  double a = dis(gen);
  double b = dis(gen);

  if (a > b) std::swap(a, b);

  if (a == b) b += 1.0;

  int num_points = 100000;
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_points));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  bessonov_e_integration_monte_carlo_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_points));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));

    bessonov_e_integration_monte_carlo_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_NEAR(reference_result[0], global_result[0], 2e1);
  }
}