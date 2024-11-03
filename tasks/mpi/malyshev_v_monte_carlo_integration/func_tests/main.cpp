#include <gtest/gtest.h>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <random>
#include <vector>
#include <iostream>

#include "mpi/malyshev_v_monte_carlo_integration/include/ops_mpi.hpp"

TEST(malyshev_v_monte_carlo_integration_mpi, test_large_random_points) {
  boost::mpi::communicator world;
  std::vector<double> global_points;
  double global_result = 0.0;
  const int num_points = 1000000;
    
  std::shared_ptr<malyshev_v_monte_carlo_integration::TaskData> taskDataPar = 
      std::make_shared<malyshev_v_monte_carlo_integration::TaskData>();

  if (world.rank() == 0) {
      global_points.resize(num_points);
      std::random_device rd;
      std::mt19937 eng(rd());
      std::uniform_real_distribution<double> distr(0.0, 1.0);

      std::generate(global_points.begin(), global_points.end(), [&]() { return distr(eng); });
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_points.data()));
      taskDataPar->inputs_count.emplace_back(global_points.size() * sizeof(double));
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_result));
      taskDataPar->outputs_count.emplace_back(sizeof(global_result));
  }

  boost::mpi::broadcast(world, global_points, 0);

  malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
      double reference_result = 0.0;
      std::shared_ptr<malyshev_v_monte_carlo_integration::TaskData> taskDataSeq = 
          std::make_shared<malyshev_v_monte_carlo_integration::TaskData>();
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_points.data()));
      taskDataSeq->inputs_count.emplace_back(global_points.size() * sizeof(double));
      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&reference_result));
      taskDataSeq->outputs_count.emplace_back(sizeof(reference_result));

      malyshev_v_monte_carlo_integration::MonteCarloIntegrationSequential testMpiTaskSequential(taskDataSeq);
      ASSERT_TRUE(testMpiTaskSequential.validation());
      testMpiTaskSequential.pre_processing();
      testMpiTaskSequential.run();
      testMpiTaskSequential.post_processing();

      ASSERT_NEAR(reference_result, global_result, 1e-5);
  }
}

TEST(malyshev_v_monte_carlo_integration_mpi, test_empty_points) {
  boost::mpi::communicator world;
  std::vector<double> global_points;
  double global_result = 0.0;
  std::shared_ptr<malyshev_v_monte_carlo_integration::TaskData> taskDataPar = 
      std::make_shared<malyshev_v_monte_carlo_integration::TaskData>();

  if (world.rank() == 0) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_points.data()));
      taskDataPar->inputs_count.emplace_back(0);
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_result));
      taskDataPar->outputs_count.emplace_back(sizeof(global_result));
  }

  boost::mpi::broadcast(world, global_points, 0);

  malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
      ASSERT_EQ(global_result, 0.0);
  }
}

TEST(malyshev_v_monte_carlo_integration_mpi, test_single_position_points) {
  boost::mpi::communicator world;
  std::vector<double> global_points(1000, 0.5);
  double global_result = 0.0;
  std::shared_ptr<malyshev_v_monte_carlo_integration::TaskData> taskDataPar = 
      std::make_shared<malyshev_v_monte_carlo_integration::TaskData>();

  if (world.rank() == 0) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_points.data()));
      taskDataPar->inputs_count.emplace_back(global_points.size() * sizeof(double));
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_result));
      taskDataPar->outputs_count.emplace_back(sizeof(global_result));
  }

  boost::mpi::broadcast(world, global_points, 0);

  malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
      double reference_result = 0.0;
      std::shared_ptr<malyshev_v_monte_carlo_integration::TaskData> taskDataSeq = 
          std::make_shared<malyshev_v_monte_carlo_integration::TaskData>();
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_points.data()));
      taskDataSeq->inputs_count.emplace_back(global_points.size() * sizeof(double));
      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&reference_result));
      taskDataSeq->outputs_count.emplace_back(sizeof(reference_result));

      malyshev_v_monte_carlo_integration::MonteCarloIntegrationSequential testMpiTaskSequential(taskDataSeq);
      ASSERT_TRUE(testMpiTaskSequential.validation());
      testMpiTaskSequential.pre_processing();
      testMpiTaskSequential.run();
      testMpiTaskSequential.post_processing();

      ASSERT_NEAR(reference_result, global_result, 1e-5);
  }
}