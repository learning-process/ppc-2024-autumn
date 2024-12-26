#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/timer.hpp>
#include <functional>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/somov_i_bitwise_sorting_batcher_merge/include/ops_mpi.hpp"

namespace somov_i_bitwise_sorting_batcher_merge_mpi {

std::vector<double> create_random_vector(int size, double mean = 3.0, double stddev = 300.0) {
  std::normal_distribution<double> norm_dist(mean, stddev);

  std::random_device rand_dev;
  std::mt19937 rand_engine(rand_dev());

  std::vector<double> tmp(size);
  for (int i = 0; i < size; i++) {
    tmp[i] = norm_dist(rand_engine);
  }
  return tmp;
}

}  // namespace somov_i_bitwise_sorting_batcher_merge_mpi

TEST(somov_i_bitwise_sorting_batcher_merge_perf_test, test_pipeline_run_10000000_elements) {
  boost::mpi::communicator world;
  std::vector<double> in = somov_i_bitwise_sorting_batcher_merge_mpi::create_random_vector(10000000);
  std::vector<double> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(somov_i_bitwise_sorting_batcher_merge_perf_test, test_task_run_10000000_elements) {
  boost::mpi::communicator world;
  std::vector<double> in = somov_i_bitwise_sorting_batcher_merge_mpi::create_random_vector(10000000);
  std::vector<double> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}
