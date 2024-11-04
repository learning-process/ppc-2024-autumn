#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/malyshev_v_monte_carlo_integration/include/ops_mpi.hpp"

TEST(malyshev_v_monte_carlo_integration_mpi, HighSampleCountPerfTest) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  double a = 0.0;
  double b = 1.0;
  double epsilon = 0.0004;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  auto testTask = std::make_shared<malyshev_v_monte_carlo_integration::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testTask->validation(), true);

  boost::mpi::timer timer;
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();
  double duration = timer.elapsed();

  if (world.rank() == 0) {
    std::cout << "HighSampleCountPerfTest duration: " << duration << " seconds\n";
  }
}

TEST(malyshev_v_monte_carlo_integration_mpi, MinimalProcessPerfTest) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  double a = 0.0;
  double b = 1.0;
  double epsilon = 0.0004;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  auto testTask = std::make_shared<malyshev_v_monte_carlo_integration::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testTask->validation(), true);

  boost::mpi::timer timer;
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();
  double duration = timer.elapsed();

  if (world.rank() == 0) {
    std::cout << "MinimalProcessPerfTest duration: " << duration << " seconds\n";
  }
}

TEST(malyshev_v_monte_carlo_integration_mpi, ExtendedRangePerfTest) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  double a = 0.0;
  double b = 10.0;
  double epsilon = 0.0004;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  auto testTask = std::make_shared<malyshev_v_monte_carlo_integration::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testTask->validation(), true);

  boost::mpi::timer timer;
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();
  double duration = timer.elapsed();

  if (world.rank() == 0) {
    std::cout << "ExtendedRangePerfTest duration: " << duration << " seconds\n";
  }
}

TEST(malyshev_v_monte_carlo_integration_mpi, SmallEpsilonPerfTest) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  double a = 0.0;
  double b = 1.0;
  double epsilon = 0.0004;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  auto testTask = std::make_shared<malyshev_v_monte_carlo_integration::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testTask->validation(), true);

  boost::mpi::timer timer;
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();
  double duration = timer.elapsed();

  if (world.rank() == 0) {
    std::cout << "SmallEpsilonPerfTest duration: " << duration << " seconds\n";
  }
}

TEST(malyshev_v_monte_carlo_integration_mpi, VaryingProcessCountPerfTest) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  double a = 0.0;
  double b = 1.0;
  double epsilon = 0.0004;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  auto testTask = std::make_shared<malyshev_v_monte_carlo_integration::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testTask->validation(), true);

  boost::mpi::timer timer;
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();
  double duration = timer.elapsed();

  if (world.rank() == 0) {
    std::cout << "VaryingProcessCountPerfTest duration: " << duration << " seconds\n";
  }
}
