#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <cmath>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/vershinina_a_integration_the_monte_carlo_method/include/ops_mpi.hpp"

TEST(vershinina_a_integration_the_monte_carlo_method, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<double> in;
  std::vector<double> global_res(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double count_size_vector;

  if (world.rank() == 0) {
    count_size_vector = 4;
    in = std::vector<double>(count_size_vector, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<vershinina_a_integration_the_monte_carlo_method::TestMPITaskParallel>(taskDataPar);
  testMpiTaskParallel->p = [](double x) { return exp(sin(4 * x) + 2 * pow(x, 2)); };
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(count_size_vector, global_res[0]);
  }
}

TEST(vershinina_a_integration_the_monte_carlo_method, test_task_run) {
  boost::mpi::communicator world;
  std::vector<double> in;
  std::vector<double> global_res(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double count_size_vector;

  if (world.rank() == 0) {
    count_size_vector = 4;
    in = std::vector<double>(count_size_vector, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<vershinina_a_integration_the_monte_carlo_method::TestMPITaskParallel>(taskDataPar);
  testMpiTaskParallel->p = [](double x) { return exp(sin(4 * x) + 2 * pow(x, 2)); };
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(count_size_vector, global_res[0]);
  }
}