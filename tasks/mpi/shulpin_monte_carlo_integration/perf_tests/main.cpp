#include <gtest/gtest.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <boost/mpi/timer.hpp>
#include <cmath>

#include "core/perf/include/perf.hpp"
#include "mpi/shulpin_monte_carlo_integration/include/monte_carlo_integral.hpp"

constexpr double ESTIMATE = 1e-3;

TEST(shulpin_monte_carlo_integration, test_pipeline_run) {
  boost::mpi::communicator world;
  double a = 0.0;
  double b = M_PI;
  int N = 1000000;
  double output = 0.0;

  std::shared_ptr<ppc::core::TaskData> task_Data_parallel = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_Data_parallel->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
    task_Data_parallel->inputs_count.push_back(1);
    task_Data_parallel->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
    task_Data_parallel->inputs_count.push_back(1);
    task_Data_parallel->inputs.push_back(reinterpret_cast<uint8_t*>(&N));
    task_Data_parallel->inputs_count.push_back(1);
    task_Data_parallel->outputs.push_back(reinterpret_cast<uint8_t*>(&output));
    task_Data_parallel->outputs_count.push_back(1);
  }

  auto test_MPI_parallel = std::make_shared<shulpin_monte_carlo_integration::TestMPITaskParallel>(task_Data_parallel);
  test_MPI_parallel->set_MPI(shulpin_monte_carlo_integration::fsin);

  ASSERT_TRUE(test_MPI_parallel->validation());
  test_MPI_parallel->pre_processing();
  test_MPI_parallel->run();
  test_MPI_parallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test_MPI_parallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    double exact = 2.0;
    EXPECT_NEAR(output, exact, ESTIMATE);
  }
}

TEST(shulpin_monte_carlo_integration, test_task_run) {
  boost::mpi::communicator world;
  double a = 0.0;
  double b = M_PI;
  int N = 1000000;
  double output = 0.0;

  std::shared_ptr<ppc::core::TaskData> task_Data_parallel = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_Data_parallel->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
    task_Data_parallel->inputs_count.push_back(1);
    task_Data_parallel->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
    task_Data_parallel->inputs_count.push_back(1);
    task_Data_parallel->inputs.push_back(reinterpret_cast<uint8_t*>(&N));
    task_Data_parallel->inputs_count.push_back(1);
    task_Data_parallel->outputs.push_back(reinterpret_cast<uint8_t*>(&output));
    task_Data_parallel->outputs_count.push_back(1);
  }

  auto test_MPI_parallel = std::make_shared<shulpin_monte_carlo_integration::TestMPITaskParallel>(task_Data_parallel);
  test_MPI_parallel->set_MPI(shulpin_monte_carlo_integration::fsin);

  ASSERT_TRUE(test_MPI_parallel->validation());
  test_MPI_parallel->pre_processing();
  test_MPI_parallel->run();
  test_MPI_parallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test_MPI_parallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    double exact = 2.0;
    EXPECT_NEAR(output, exact, ESTIMATE);
  }
}