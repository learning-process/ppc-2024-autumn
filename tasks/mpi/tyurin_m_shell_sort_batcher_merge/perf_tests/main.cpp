#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <cmath>
#include <random>

#include "core/perf/include/perf.hpp"
#include "mpi/tyurin_m_shell_sort_batcher_merge/include/ops_mpi.hpp"

namespace tyurin_m_shell_sort_batcher_merge_mpi {

std::vector<int> random_vector(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(-100, 100);

  std::vector<int> arr(size);
  for (size_t i = 0; i < arr.size(); i++) {
    arr[i] = dist(gen);
  }
  return arr;
}

}  // namespace tyurin_m_shell_sort_batcher_merge_mpi

TEST(tyurin_m_shell_sort_batcher_merge_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  if (world.size() > 1 && world.size() % 2 != 0) GTEST_SKIP();
  const int n = std::pow(2, 15);
  std::vector<int> input_vec;
  std::vector<int> output_vec;

  auto task = std::make_shared<ppc::core::TaskData>();

  task->inputs_count.emplace_back(n);
  if (world.rank() == 0) {
    input_vec = tyurin_m_shell_sort_batcher_merge_mpi::random_vector(n);
    output_vec.resize(n);

    task->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vec.data()));
    task->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vec.data()));
  }

  auto test = std::make_shared<tyurin_m_shell_sort_batcher_merge_mpi::ShellSortBatcherMerge>(task);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    std::vector<int> exp_result = input_vec;
    std::sort(exp_result.begin(), exp_result.end());
    EXPECT_EQ(exp_result, output_vec);
  }
}

TEST(tyurin_m_shell_sort_batcher_merge_mpi, test_task_run) {
  boost::mpi::communicator world;
  if (world.size() > 1 && world.size() % 2 != 0) GTEST_SKIP();
  const int n = std::pow(2, 15);
  std::vector<int> input_vec;
  std::vector<int> output_vec;

  auto task = std::make_shared<ppc::core::TaskData>();

  task->inputs_count.emplace_back(n);
  if (world.rank() == 0) {
    input_vec = tyurin_m_shell_sort_batcher_merge_mpi::random_vector(n);
    output_vec.resize(n);

    task->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vec.data()));
    task->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vec.data()));
  }

  auto test = std::make_shared<tyurin_m_shell_sort_batcher_merge_mpi::ShellSortBatcherMerge>(task);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    std::vector<int> exp_result = input_vec;
    std::sort(exp_result.begin(), exp_result.end());
    EXPECT_EQ(exp_result, output_vec);
  }
}
