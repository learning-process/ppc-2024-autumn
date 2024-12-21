#include <gtest/gtest.h>

#include <boost/mpi.hpp>
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

TEST(MonteCarloIntegrationTaskParallel, Test_Work_Distribution) {
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

    int num_processes = world.size();
    int num_points = 1000000;
    int local_num_points = num_points / num_processes;

    std::vector<int> local_points_count(num_processes, 0);
    boost::mpi::all_gather(world, local_num_points, local_points_count);

    for (int i = 0; i < num_processes; ++i) {
      ASSERT_EQ(local_points_count[i], local_num_points);
    }
  } catch (const std::exception& e) {
    std::cerr << "Process " << world.rank() << " caught exception: " << e.what() << std::endl;
    throw;
  }
}

TEST(MonteCarloIntegrationTaskParallel, Test_Uneven_Points_Distribution) {
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

    int num_processes = world.size();
    int rank = world.rank();
    int num_points = 1000000;
    int local_num_points = num_points / num_processes + (rank < num_points % num_processes ? 1 : 0);

    std::vector<int> local_points_count(num_processes, 0);
    boost::mpi::all_gather(world, local_num_points, local_points_count);

    int total_points = 0;
    for (int i = 0; i < num_processes; ++i) {
      total_points += local_points_count[i];
    }
    ASSERT_EQ(total_points, num_points);
  } catch (const std::exception& e) {
    std::cerr << "Process " << world.rank() << " caught exception: " << e.what() << std::endl;
    throw;
  }
}

TEST(MonteCarloIntegrationTaskParallel, Test_Validation_Failure) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  try {
    std::vector<double> out(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
      taskDataPar->inputs_count.emplace_back(1);
      taskDataPar->outputs.emplace_back(nullptr);
      taskDataPar->outputs_count.emplace_back(1);
    }

    auto testMpiTaskParallel =
        std::make_shared<shuravina_o_monte_carlo::MonteCarloIntegrationTaskParallel>(taskDataPar);

    ASSERT_EQ(testMpiTaskParallel->validation(), false);
  } catch (const std::exception& e) {
    std::cerr << "Process " << world.rank() << " caught exception: " << e.what() << std::endl;
    throw;
  }
}
TEST(MonteCarloIntegrationTaskParallel, Test_Zero_Points) {
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
    testMpiTaskParallel->set_num_points(0);

    ASSERT_EQ(testMpiTaskParallel->validation(), true);
    testMpiTaskParallel->pre_processing();
    testMpiTaskParallel->run();
    testMpiTaskParallel->post_processing();

    if (world.rank() == 0) {
      ASSERT_NEAR(0.0, out[0], 0.01);
    }
  } catch (const std::exception& e) {
    std::cerr << "Process " << world.rank() << " caught exception: " << e.what() << std::endl;
    throw;
  }
}