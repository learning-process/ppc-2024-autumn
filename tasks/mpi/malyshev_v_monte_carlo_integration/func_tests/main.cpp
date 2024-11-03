#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <cmath>
#include <memory>

#include "mpi/malyshev_v_monte_carlo_integration/include/ops_mpi.hpp"

using namespace malyshev_v_monte_carlo_integration;

constexpr double expected_integral_const = 1.0;
constexpr double expected_integral_linear = 0.5;
constexpr double expected_integral_quad = 1.0 / 3.0;
constexpr double expected_integral_sin = 1.0 - std::cos(1.0);
constexpr double expected_integral_exp = std::exp(1.0) - 1.0;

const double tolerance = 0.01;

template <typename TaskType>
void run_monte_carlo_test(const std::function<double(double)>& func, double a, double b, int n,
                          double expected_result) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  double result = 0.0;
  
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  taskData->outputs_count.emplace_back(1);

  TaskType task(taskData);
  task.set_function(func);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  EXPECT_NEAR(result, expected_result, tolerance);
}

TEST(MonteCarloIntegrationTests, Test_ConstantFunction) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    run_monte_carlo_test<MonteCarloIntegrationParallel>([](double x) { return 1.0; }, 0.0, 1.0, 100000,
                                                        expected_integral_const);
  }
}

TEST(MonteCarloIntegrationTests, Test_LinearFunction) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    run_monte_carlo_test<MonteCarloIntegrationParallel>([](double x) { return x; }, 0.0, 1.0, 100000,
                                                        expected_integral_linear);
  }
}

TEST(MonteCarloIntegrationTests, Test_QuadraticFunction) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    run_monte_carlo_test<MonteCarloIntegrationParallel>([](double x) { return x * x; }, 0.0, 1.0, 100000,
                                                        expected_integral_quad);
  }
}

TEST(MonteCarloIntegrationTests, Test_SineFunction) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    run_monte_carlo_test<MonteCarloIntegrationParallel>([](double x) { return std::sin(x); }, 0.0, 1.0, 100000,
                                                        expected_integral_sin);
  }
}

TEST(MonteCarloIntegrationTests, Test_ExponentialFunction) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    run_monte_carlo_test<MonteCarloIntegrationParallel>([](double x) { return std::exp(x); }, 0.0, 1.0, 100000,
                                                        expected_integral_exp);
  }
}

int main(int argc, char** argv) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (world.rank() != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }
  return RUN_ALL_TESTS();
}
