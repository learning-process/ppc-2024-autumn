#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/malyshev_v_monte_carlo_integration/include/ops_mpi.hpp"

TEST(malyshev_v_monte_carlo_integration_mpi, Test_Integration_mpi_SquareFunction) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  double a = 0.0;
  double b = 1.0;
  double epsilon = 0.0004;

  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  malyshev_v_monte_carlo_integration::TestMPITaskParallel testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    double expected_value = 0.3333;
    ASSERT_NEAR(global_result[0], expected_value, epsilon);
  }
}

TEST(malyshev_v_monte_carlo_integration_mpi, Test_Integration_mpi_ConstantFunction) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  double a = 0.0;
  double b = 1.0;
  double epsilon = 0.0004;

  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  malyshev_v_monte_carlo_integration::TestMPITaskParallel testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    double expected_value = 1.0;
    ASSERT_NEAR(global_result[0], expected_value, epsilon);
  }
}

TEST(malyshev_v_monte_carlo_integration_mpi, Test_Integration_mpi_LargeInterval) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  double a = 0.0;
  double b = 10.0;
  double epsilon = 0.0004;

  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  malyshev_v_monte_carlo_integration::TestMPITaskParallel testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    double expected_value = 333.3;
    ASSERT_NEAR(global_result[0], expected_value, epsilon * 10);
  }
}

TEST(malyshev_v_monte_carlo_integration_mpi, Test_Integration_mpi_NegativeRange) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  double a = -1.0;
  double b = 1.0;
  double epsilon = 0.0004;

  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  malyshev_v_monte_carlo_integration::TestMPITaskParallel testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    double expected_value = 2.0 / 3.0;
    ASSERT_NEAR(global_result[0], expected_value, epsilon);
  }
}

TEST(malyshev_v_monte_carlo_integration_mpi, Test_Integration_mpi_ZeroWidthInterval) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  double a = 1.0;
  double b = 1.0;
  double epsilon = 0.0004;

  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  malyshev_v_monte_carlo_integration::TestMPITaskParallel testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    double expected_value = 0.0;
    ASSERT_NEAR(global_result[0], expected_value, epsilon);
  }
}
