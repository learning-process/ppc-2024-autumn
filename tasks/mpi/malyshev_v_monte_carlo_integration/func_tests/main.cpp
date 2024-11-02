#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <random>

#include "mpi/malyshev_v_monte_carlo_integration/include/malyshev_v_monte_carlo_integration.hpp"  

TEST(malyshev_v_monte_carlo_integration, Test_Integration_mpi_1) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double a = 0.0;
  double b = 1.0;
  int n_samples = 1000000;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n_samples));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  malyshev_v_monte_carlo_integration::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n_samples));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));

    malyshev_v_monte_carlo_integration::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_NEAR(reference_result[0], global_result[0], 1e-1);
  }
}

TEST(malyshev_v_monte_carlo_integration, Test_Integration_mpi_2) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double a = 1.0;
  double b = 2.0;
  int n_samples = 500000;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n_samples));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  malyshev_v_monte_carlo_integration::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n_samples));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));

    malyshev_v_monte_carlo_integration::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_NEAR(reference_result[0], global_result[0], 1e-1);
  }
}

TEST(malyshev_v_monte_carlo_integration, Test_Integration_mpi_random) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::random_device dev;
  std::mt19937 gen(dev());
  double a = (gen() % 100) / 100.0;
  double b = (gen() % 100) / 100.0;
  if (a == b) b += 0.1;  // Ensure a and b are not equal
  int n_samples = 100000;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n_samples));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  malyshev_v_monte_carlo_integration::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n_samples));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));

    malyshev_v_monte_carlo_integration::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_NEAR(reference_result[0], global_result[0], 1e-1);
  }
}

TEST(malyshev_v_monte_carlo_integration, TaskMpi_InputSizeLessThan3) {
  std::shared_ptr<ppc::core::TaskData> taskDataMPIParallel = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    double a = -1.0;
    double b = 1.0;
    taskDataMPIParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataMPIParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    // Недостаточно входных параметров
    malyshev_v_monte_carlo_integration::TestMPITaskParallel testTaskMPIParallel(taskDataMPIParallel);
    ASSERT_EQ(testTaskMPIParallel.validation(), false);
  }
}

TEST(malyshev_v_monte_carlo_integration, TaskMpi_OutputSizeMoreThan1) {
  std::shared_ptr<ppc::core::TaskData> taskDataMPIParallel = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    double a = -1.0;
    double b = 1.0;
    int n_samples = 100000;
    taskDataMPIParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataMPIParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataMPIParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n_samples));
    // Более одного выходного параметра
    double result1 = 0.0;
    double result2 = 0.0;
    taskDataMPIParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result1));
    taskDataMPIParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result2));
    malyshev_v_monte_carlo_integration::TestMPITaskParallel testTaskMPIParallel(taskDataMPIParallel);
    ASSERT_EQ(testTaskMPIParallel.validation(), false);
  }
}
