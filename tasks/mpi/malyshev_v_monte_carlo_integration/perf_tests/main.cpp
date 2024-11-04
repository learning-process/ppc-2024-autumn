#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <chrono>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/malyshev_v_monte_carlo_integration/include/ops_mpi.hpp"

TEST(LargeSample_SquareFunction, {
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  double a = 0.0;
  double b = 1.0;
  double epsilon = 0.0004;

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(new double[1]));

  malyshev_v_monte_carlo_integration::TestMPITaskParallel testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
})

TEST(LargeInterval_SquareFunction, {
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  double a = 0.0;
  double b = 100.0;
  double epsilon = 0.0004;

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(new double[1]));

  malyshev_v_monte_carlo_integration::TestMPITaskParallel testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
})

TEST(CubicFunction_LargeInterval, {
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  double a = 0.0;
  double b = 10.0;
  double epsilon = 0.0004;

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(new double[1]));

  malyshev_v_monte_carlo_integration::TestMPITaskParallel testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
})

TEST(HighAccuracy_ConstantFunction, {
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  double a = 0.0;
  double b = 1.0;
  double epsilon = 0.0004;

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(new double[1]));

  malyshev_v_monte_carlo_integration::TestMPITaskParallel testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
})
