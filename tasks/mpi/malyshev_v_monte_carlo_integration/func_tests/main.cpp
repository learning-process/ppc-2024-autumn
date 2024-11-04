#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/malyshev_v_monte_carlo_integration/include/ops_mpi.hpp"

TEST(malyshev_v_monte_carlo_integration_mpi, BasicIntegrationTest) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  double a = 0.0;
  double b = 1.0;
  double epsilon = 0.0004;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  malyshev_v_monte_carlo_integration::TestMPITaskParallel testTask(taskDataPar);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    double expected_value = 0.3333;
    ASSERT_NEAR(global_result[0], expected_value, epsilon);
  }
}

TEST(malyshev_v_monte_carlo_integration_mpi, LargeRangeIntegrationTest) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  double a = 0.0;
  double b = 10.0;
  double epsilon = 0.001;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  malyshev_v_monte_carlo_integration::TestMPITaskParallel testTask(taskDataPar);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    double expected_value = 333.3;
    ASSERT_NEAR(global_result[0], expected_value, epsilon);
  }
}

TEST(malyshev_v_monte_carlo_integration_mpi, SmallRangeIntegrationTest) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  double a = 0.0;
  double b = 0.1;
  double epsilon = 0.0001;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  malyshev_v_monte_carlo_integration::TestMPITaskParallel testTask(taskDataPar);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    double expected_value = 0.0003333;
    ASSERT_NEAR(global_result[0], expected_value, epsilon);
  }
}

TEST(malyshev_v_monte_carlo_integration_mpi, ZeroRangeIntegrationTest) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  double a = 1.0;
  double b = 1.0;  // Zero range
  double epsilon = 0.0001;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  malyshev_v_monte_carlo_integration::TestMPITaskParallel testTask(taskDataPar);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    double expected_value = 0.0;
    ASSERT_NEAR(global_result[0], expected_value, epsilon);
  }
}

TEST(malyshev_v_monte_carlo_integration_mpi, NegativeRangeIntegrationTest) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  double a = -1.0;
  double b = 0.0;
  double epsilon = 0.0004;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  malyshev_v_monte_carlo_integration::TestMPITaskParallel testTask(taskDataPar);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    double expected_value = 0.3333;
    ASSERT_NEAR(global_result[0], expected_value, epsilon);
  }
}
