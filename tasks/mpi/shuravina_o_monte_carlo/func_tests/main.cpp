#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <numbers>
#include <vector>

#include "mpi/shuravina_o_monte_carlo/include/ops_mpi.hpp"

TEST(MonteCarloIntegrationTaskParallel, Test_Integration) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  try {
    std::vector<double> out(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
      taskDataPar->inputs.emplace_back(nullptr);
      taskDataPar->inputs_count.emplace_back(0);
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
      taskDataPar->outputs_count.emplace_back(out.size());
    }

    auto testMpiTaskParallel =
        std::make_shared<shuravina_o_monte_carlo::MonteCarloIntegrationTaskParallel>(taskDataPar);
    testMpiTaskParallel->set_interval(0.0, 1.0);
    testMpiTaskParallel->set_num_points(1000000);
    testMpiTaskParallel->set_function([](double x) { return x * x; });

    ASSERT_EQ(testMpiTaskParallel->validation(), true);
    testMpiTaskParallel->pre_processing();
    testMpiTaskParallel->run();
    testMpiTaskParallel->post_processing();

    if (world.rank() == 0) {
      double expected_integral = 1.0 / 3.0;
      ASSERT_NEAR(expected_integral, out[0], 0.01);
    }
  } catch (const std::exception& e) {
    std::cerr << "Process " << world.rank() << " caught exception: " << e.what() << std::endl;
    throw;
  }
}

TEST(MonteCarloIntegrationTaskParallel, Test_Boundary_Conditions) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  try {
    std::vector<double> out(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
      taskDataPar->inputs.emplace_back(nullptr);
      taskDataPar->inputs_count.emplace_back(0);
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
      taskDataPar->outputs_count.emplace_back(out.size());
    }

    auto testMpiTaskParallel =
        std::make_shared<shuravina_o_monte_carlo::MonteCarloIntegrationTaskParallel>(taskDataPar);
    testMpiTaskParallel->set_interval(-1.0, 1.0);
    testMpiTaskParallel->set_num_points(1000000);
    testMpiTaskParallel->set_function([](double x) { return x * x; });

    ASSERT_EQ(testMpiTaskParallel->validation(), true);
    testMpiTaskParallel->pre_processing();
    testMpiTaskParallel->run();
    testMpiTaskParallel->post_processing();

    if (world.rank() == 0) {
      double expected_integral = 2.0 / 3.0;
      ASSERT_NEAR(expected_integral, out[0], 0.01);
    }
  } catch (const std::exception& e) {
    std::cerr << "Process " << world.rank() << " caught exception: " << e.what() << std::endl;
    throw;
  }
}

TEST(MonteCarloIntegrationTaskParallel, Test_Different_Functions) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  try {
    std::vector<double> out(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
      taskDataPar->inputs.emplace_back(nullptr);
      taskDataPar->inputs_count.emplace_back(0);
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
      taskDataPar->outputs_count.emplace_back(out.size());
    }

    auto testMpiTaskParallel =
        std::make_shared<shuravina_o_monte_carlo::MonteCarloIntegrationTaskParallel>(taskDataPar);
    testMpiTaskParallel->set_interval(0.0, 1.0);
    testMpiTaskParallel->set_num_points(1000000);

    std::vector<std::function<double(double)>> functions = {
        [](double x) { return x * x; }, [](double x) { return std::sin(x); }, [](double x) { return std::exp(x); },
        [](double x) { return std::cos(x); }};

    for (const auto& func : functions) {
      testMpiTaskParallel->set_function(func);
      ASSERT_EQ(testMpiTaskParallel->validation(), true);
      testMpiTaskParallel->pre_processing();
      testMpiTaskParallel->run();
      testMpiTaskParallel->post_processing();

      if (world.rank() == 0) {
        double expected_integral = 1.0 / 3.0;
        if (func(0.5) == std::sin(0.5)) {
          expected_integral = 1.0 - std::cos(1.0);
        } else if (func(0.5) == std::exp(0.5)) {
          expected_integral = std::numbers::e - 1.0;
        } else if (func(0.5) == std::cos(0.5)) {
          expected_integral = std::sin(1.0);
        }
        ASSERT_NEAR(expected_integral, out[0], 0.01);
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Process " << world.rank() << " caught exception: " << e.what() << std::endl;
    throw;
  }
}