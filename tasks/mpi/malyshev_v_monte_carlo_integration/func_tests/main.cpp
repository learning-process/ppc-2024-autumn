#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "core/task/include/task.hpp"
#include "mpi/malyshev_v_monte_carlo_integration/include/ops_mpi.hpp"

TEST(malyshev_v_monte_carlo_integration_mpi, LinearFunctionTest) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  double a = 0.0;
  double b = 1.0;
  double epsilon = 0.0004;
  auto linear_function = [](double x) { return x; };

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  malyshev_v_monte_carlo_integration::TestMPITaskParallel testTask(taskDataPar, linear_function);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    double expected_value = 0.5;
    ASSERT_NEAR(global_result[0], expected_value, epsilon);
  }
}

TEST(malyshev_v_monte_carlo_integration_mpi, CubicFunctionTest) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  double a = 0.0;
  double b = 1.0;
  double epsilon = 0.0004;
  auto cubic_function = [](double x) { return x * x * x; };

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  malyshev_v_monte_carlo_integration::TestMPITaskParallel testTask(taskDataPar, cubic_function);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    double expected_value = 0.25;
    ASSERT_NEAR(global_result[0], expected_value, epsilon);
  }
}

TEST(malyshev_v_monte_carlo_integration_mpi, SineFunctionTest) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  double a = 0.0;
  double b = M_PI;
  double epsilon = 0.0004;
  auto sine_function = [](double x) { return std::sin(x); };

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  malyshev_v_monte_carlo_integration::TestMPITaskParallel testTask(taskDataPar, sine_function);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    double expected_value = 2.0;
    ASSERT_NEAR(global_result[0], expected_value, epsilon);
  }
}

TEST(malyshev_v_monte_carlo_integration_mpi, ExponentialFunctionTest) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  double a = 0.0;
  double b = 1.0;
  double epsilon = 0.0004;
  auto exponential_function = [](double x) { return std::exp(x); };

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  malyshev_v_monte_carlo_integration::TestMPITaskParallel testTask(taskDataPar, exponential_function);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    double expected_value = std::exp(1.0) - 1;
    ASSERT_NEAR(global_result[0], expected_value, epsilon);
  }
}
