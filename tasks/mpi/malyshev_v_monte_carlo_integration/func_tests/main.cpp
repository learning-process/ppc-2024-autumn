#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <random>

#include "mpi/malyshev_v_monte_carlo_integration/include/ops_mpi.hpp"

// Test Monte Carlo integration on a small interval
TEST(malyshev_v_monte_carlo_integration, Test_Integration_mpi_small_interval) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double a = 0.0;
  double b = 1.0;
  double epsilon = 0.001;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));

    assert(taskDataPar->inputs[0] != nullptr);
    assert(taskDataPar->inputs[1] != nullptr);
    assert(taskDataPar->inputs[2] != nullptr);

    assert(taskDataPar->outputs[0] != nullptr);
  }

  malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel MonteCarloIntegrationParallel(taskDataPar);
  ASSERT_EQ(MonteCarloIntegrationParallel.validation(), true);
  MonteCarloIntegrationParallel.pre_processing();
  MonteCarloIntegrationParallel.run();
  MonteCarloIntegrationParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    malyshev_v_monte_carlo_integration::MonteCarloIntegrationSequential MonteCarloIntegrationSequential(taskDataSeq);
    ASSERT_EQ(MonteCarloIntegrationSequential.validation(), true);
    MonteCarloIntegrationSequential.pre_processing();
    MonteCarloIntegrationSequential.run();
    MonteCarloIntegrationSequential.post_processing();
    ASSERT_NEAR(reference_result[0], global_result[0], 1e-1);
  }
}

// Test Monte Carlo integration over a large range
TEST(malyshev_v_monte_carlo_integration, Test_Integration_mpi_large_range) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double a = -100.0;
  double b = 100.0;
  double epsilon = 0.001;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));

    assert(taskDataPar->inputs[0] != nullptr);
    assert(taskDataPar->inputs[1] != nullptr);
    assert(taskDataPar->inputs[2] != nullptr);

    assert(taskDataPar->outputs[0] != nullptr);
  }

  malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel MonteCarloIntegrationParallel(taskDataPar);
  ASSERT_EQ(MonteCarloIntegrationParallel.validation(), true);
  MonteCarloIntegrationParallel.pre_processing();
  MonteCarloIntegrationParallel.run();
  MonteCarloIntegrationParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    malyshev_v_monte_carlo_integration::MonteCarloIntegrationSequential MonteCarloIntegrationSequential(taskDataSeq);
    ASSERT_EQ(MonteCarloIntegrationSequential.validation(), true);
    MonteCarloIntegrationSequential.pre_processing();
    MonteCarloIntegrationSequential.run();
    MonteCarloIntegrationSequential.post_processing();
    ASSERT_NEAR(reference_result[0], global_result[0], 1e-1);
  }
}

// Test with random interval bounds to ensure robustness
TEST(malyshev_v_monte_carlo_integration, Test_Integration_mpi_random) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(-50.0, 50.0);

  double a = dist(gen);
  double b = dist(gen);
  if (a > b) std::swap(a, b);
  double epsilon = 0.001;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));

    assert(taskDataPar->inputs[0] != nullptr);
    assert(taskDataPar->inputs[1] != nullptr);
    assert(taskDataPar->inputs[2] != nullptr);

    assert(taskDataPar->outputs[0] != nullptr);
  }

  malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel MonteCarloIntegrationParallel(taskDataPar);
  ASSERT_EQ(MonteCarloIntegrationParallel.validation(), true);
  MonteCarloIntegrationParallel.pre_processing();
  MonteCarloIntegrationParallel.run();
  MonteCarloIntegrationParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));

    assert(taskDataPar->inputs[0] != nullptr);
    assert(taskDataPar->inputs[1] != nullptr);
    assert(taskDataPar->inputs[2] != nullptr);

    assert(taskDataPar->outputs[0] != nullptr);

    malyshev_v_monte_carlo_integration::MonteCarloIntegrationSequential MonteCarloIntegrationSequential(taskDataSeq);
    ASSERT_EQ(MonteCarloIntegrationSequential.validation(), true);
    MonteCarloIntegrationSequential.pre_processing();
    MonteCarloIntegrationSequential.run();
    MonteCarloIntegrationSequential.post_processing();
    ASSERT_NEAR(reference_result[0], global_result[0], 1e-1);
  }
}

// Test input validation when the input size is incorrect
TEST(malyshev_v_monte_carlo_integration, Test_Validation_InputSizeLessThan3) {
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    double a = -1.0;
    double b = 1.0;
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    double result = 0.0;
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  }
  malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel task(taskData);
  ASSERT_FALSE(task.validation());
}

// Test input validation when there are extra inputs
TEST(malyshev_v_monte_carlo_integration, Test_Validation_ExtraInput) {
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    double a = -1.0;
    double b = 1.0;
    double epsilon = 0.001;
    double extra = 5.0;
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&extra));
    double result = 0.0;
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  }
  malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel task(taskData);
  ASSERT_FALSE(task.validation());
}
