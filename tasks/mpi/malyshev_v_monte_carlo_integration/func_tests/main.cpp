#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <vector>

#include "mpi/malyshev_v_monte_carlo_integration/include/ops_mpi.hpp"

TEST(Parallel_Operations_MPI, Test_Integral_Calculation) {
  boost::mpi::communicator world;
  std::vector<double> global_results(1, 0.0);
  std::shared_ptr<malyshev_v_monte_carlo_integration::TaskData> taskDataPar =
      std::make_shared<malyshev_v_monte_carlo_integration::TaskData>();

  if (world.rank() == 0) {
    const int count_size_samples = 5;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_size_samples));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_results.data()));
  }

  malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    double expected_result = 1.0;
    ASSERT_NEAR(global_results[0], expected_result, 0.01);
  }
}

TEST(Parallel_Operations_MPI, Test_Error_Distribution) {
  boost::mpi::communicator world;
  std::vector<double> global_errors(1, 0.0);
  std::shared_ptr<malyshev_v_monte_carlo_integration::TaskData> taskDataPar =
      std::make_shared<malyshev_v_monte_carlo_integration::TaskData>();

  if (world.rank() == 0) {
    const int count_size_samples = 5;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_size_samples));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_errors.data()));
  }

  malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_GT(global_errors[0], 0.0);
  }
}

TEST(Parallel_Operations_MPI, Test_Correlation_Parallel_Sequential) {
  boost::mpi::communicator world;
  std::vector<double> parallel_results(1, 0.0);
  std::vector<double> sequential_results(1, 0.0);
  std::shared_ptr<malyshev_v_monte_carlo_integration::TaskData> taskDataPar =
      std::make_shared<malyshev_v_monte_carlo_integration::TaskData>();
  std::shared_ptr<malyshev_v_monte_carlo_integration::TaskData> taskDataSeq =
      std::make_shared<malyshev_v_monte_carlo_integration::TaskData>();

  const int count_size_samples = 5;
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_size_samples));
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallel_results.data()));

  malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_size_samples));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sequential_results.data()));

    malyshev_v_monte_carlo_integration::MonteCarloIntegrationSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_NEAR(parallel_results[0], sequential_results[0], 0.01);
  }
}

TEST(Parallel_Operations_MPI, Test_Performance) {
  boost::mpi::communicator world;
  std::vector<double> global_results(1, 0.0);
  std::shared_ptr<malyshev_v_monte_carlo_integration::TaskData> taskDataPar =
      std::make_shared<malyshev_v_monte_carlo_integration::TaskData>();

  if (world.rank() == 0) {
    const int count_size_samples = 1000000;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_size_samples));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_results.data()));
  }

  malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  auto start = std::chrono::high_resolution_clock::now();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  if (world.rank() == 0) {
    ASSERT_LT(duration, 1000);
  }
}

TEST(Parallel_Operations_MPI, Test_Integral_Exact) {
  boost::mpi::communicator world;
  std::vector<double> global_results(1, 0.0);
  std::shared_ptr<malyshev_v_monte_carlo_integration::TaskData> taskDataPar =
      std::make_shared<malyshev_v_monte_carlo_integration::TaskData>();

  if (world.rank() == 0) {
    const int count_size_samples = 5;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_size_samples));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_results.data()));
  }

  malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    double expected_result = 1.0;
    ASSERT_NEAR(global_results[0], expected_result, 0.01);
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
