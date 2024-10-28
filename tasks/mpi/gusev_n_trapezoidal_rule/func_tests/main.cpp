#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <vector>

#include "mpi/gusev_n_trapezoidal_rule/include/ops_mpi.hpp"

TEST(gusev_n_trapezoidal_rule_mpi, ConstantFunctionTest) {
  boost::mpi::communicator world;
  std::vector<double> result_global(1, 0);

  auto taskDataParallel = std::make_shared<ppc::core::TaskData>();

  double lower_bound = 0.0;
  double upper_bound = 10.0;
  int intervals = 1000000;

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
    taskDataParallel->inputs_count.emplace_back(1);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
    taskDataParallel->inputs_count.emplace_back(1);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&intervals));
    taskDataParallel->inputs_count.emplace_back(1);
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_global.data()));
    taskDataParallel->outputs_count.emplace_back(result_global.size());
  }

  gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationParallel parallelTask(taskDataParallel);
  parallelTask.set_function([](double x) { return 5.0; });
  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0);

    auto taskDataSequential = std::make_shared<ppc::core::TaskData>();
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
    taskDataSequential->inputs_count.emplace_back(1);
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
    taskDataSequential->inputs_count.emplace_back(1);
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(&intervals));
    taskDataSequential->inputs_count.emplace_back(1);
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSequential->outputs_count.emplace_back(reference_result.size());

    gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationSequential sequentialTask(taskDataSequential);
    sequentialTask.set_function([](double x) { return 5.0; });
    ASSERT_EQ(sequentialTask.validation(), true);
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    ASSERT_NEAR(reference_result[0], result_global[0], 1e-3);
  }
}

TEST(gusev_n_trapezoidal_rule_mpi, SquareFunctionTest) {
  boost::mpi::communicator world;
  std::vector<double> result_global(1, 0);
  auto taskDataParallel = std::make_shared<ppc::core::TaskData>();

  double lower_bound = 0.0;
  double upper_bound = 5.0;
  int intervals = 1000000;

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
    taskDataParallel->inputs_count.emplace_back(1);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
    taskDataParallel->inputs_count.emplace_back(1);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&intervals));
    taskDataParallel->inputs_count.emplace_back(1);
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_global.data()));
    taskDataParallel->outputs_count.emplace_back(result_global.size());
  }

  gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationParallel parallelTask(taskDataParallel);
  parallelTask.set_function([](double x) { return x * x; });
  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0);
    auto taskDataSequential = std::make_shared<ppc::core::TaskData>();
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
    taskDataSequential->inputs_count.emplace_back(1);
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
    taskDataSequential->inputs_count.emplace_back(1);
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(&intervals));
    taskDataSequential->inputs_count.emplace_back(1);
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSequential->outputs_count.emplace_back(reference_result.size());

    gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationSequential sequentialTask(taskDataSequential);
    sequentialTask.set_function([](double x) { return x * x; });
    ASSERT_EQ(sequentialTask.validation(), true);
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    ASSERT_NEAR(reference_result[0], result_global[0], 1e-3);
  }
}

TEST(gusev_n_trapezoidal_rule_mpi, SineFunctionTest) {
  boost::mpi::communicator world;
  std::vector<double> result_global(1, 0);
  auto taskDataParallel = std::make_shared<ppc::core::TaskData>();

  double lower_bound = 0.0;
  double upper_bound = M_PI;
  int intervals = 1000000;

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
    taskDataParallel->inputs_count.emplace_back(1);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
    taskDataParallel->inputs_count.emplace_back(1);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&intervals));
    taskDataParallel->inputs_count.emplace_back(1);
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_global.data()));
    taskDataParallel->outputs_count.emplace_back(result_global.size());
  }

  gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationParallel parallelTask(taskDataParallel);
  parallelTask.set_function([](double x) { return std::sin(x); });
  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0);
    auto taskDataSequential = std::make_shared<ppc::core::TaskData>();
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
    taskDataSequential->inputs_count.emplace_back(1);
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
    taskDataSequential->inputs_count.emplace_back(1);
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(&intervals));
    taskDataSequential->inputs_count.emplace_back(1);
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSequential->outputs_count.emplace_back(reference_result.size());

    gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationSequential sequentialTask(taskDataSequential);
    sequentialTask.set_function([](double x) { return std::sin(x); });
    ASSERT_EQ(sequentialTask.validation(), true);
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    ASSERT_NEAR(reference_result[0], result_global[0], 1e-3);
  }
}

TEST(gusev_n_trapezoidal_rule_mpi, ExponentialFunctionTest) {
  boost::mpi::communicator world;
  std::vector<double> result_global(1, 0);
  auto taskDataParallel = std::make_shared<ppc::core::TaskData>();

  double lower_bound = 0.0;
  double upper_bound = 1.0;
  int intervals = 1000000;

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
    taskDataParallel->inputs_count.emplace_back(1);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
    taskDataParallel->inputs_count.emplace_back(1);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&intervals));
    taskDataParallel->inputs_count.emplace_back(1);
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_global.data()));
    taskDataParallel->outputs_count.emplace_back(result_global.size());
  }

  gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationParallel parallelTask(taskDataParallel);
  parallelTask.set_function([](double x) { return std::exp(x); });
  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0);
    auto taskDataSequential = std::make_shared<ppc::core::TaskData>();
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
    taskDataSequential->inputs_count.emplace_back(1);
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
    taskDataSequential->inputs_count.emplace_back(1);
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(&intervals));
    taskDataSequential->inputs_count.emplace_back(1);
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSequential->outputs_count.emplace_back(reference_result.size());

    gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationSequential sequentialTask(taskDataSequential);
    sequentialTask.set_function([](double x) { return std::exp(x); });
    ASSERT_EQ(sequentialTask.validation(), true);
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    ASSERT_NEAR(reference_result[0], result_global[0], 1e-3);
  }
}

TEST(gusev_n_trapezoidal_rule_mpi, RemainderCaseTest) {
  boost::mpi::communicator world;
  std::vector<double> result_global(1, 0);
  auto taskDataParallel = std::make_shared<ppc::core::TaskData>();

  double lower_bound = 0.0;
  double upper_bound = 5.0;
  int intervals = 1000;

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
    taskDataParallel->inputs_count.emplace_back(1);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
    taskDataParallel->inputs_count.emplace_back(1);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&intervals));
    taskDataParallel->inputs_count.emplace_back(1);
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_global.data()));
    taskDataParallel->outputs_count.emplace_back(result_global.size());
  }

  gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationParallel parallelTask(taskDataParallel);
  parallelTask.set_function([](double x) { return x * x; });
  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0);
    auto taskDataSequential = std::make_shared<ppc::core::TaskData>();
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
    taskDataSequential->inputs_count.emplace_back(1);
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
    taskDataSequential->inputs_count.emplace_back(1);
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(&intervals));
    taskDataSequential->inputs_count.emplace_back(1);
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSequential->outputs_count.emplace_back(reference_result.size());

    gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationSequential sequentialTask(taskDataSequential);
    sequentialTask.set_function([](double x) { return x * x; });
    ASSERT_EQ(sequentialTask.validation(), true);
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    ASSERT_NEAR(reference_result[0], result_global[0], 1e-3);
  }
}
