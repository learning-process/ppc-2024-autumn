#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/gordeeva_t_shell_sort_batcher_merge/include/ops_mpi.hpp"

std::vector<int> rand_vec(int size, int down = -100, int upp = 100) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(down, upp);

  std::vector<int> v(size);
  for (auto &number : v) {
    number = dis(gen);
  }
  return v;
}

TEST(gordeeva_t_shell_sort_batcher_merge_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  const int size = 3000000;

  std::vector<int> input_values;
  std::vector<int> output_values(size, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_values = rand_vec(size, 0, 1000);
  }

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_values.data()));
    taskDataPar->inputs_count.emplace_back(static_cast<unsigned int>(input_values.size()));

    output_values.resize(input_values.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_values.data()));
    taskDataPar->outputs_count.emplace_back(static_cast<unsigned int>(output_values.size()));
  }

  auto testMpiTaskParallel =
      std::make_shared<gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel>(taskDataPar);

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

    ASSERT_TRUE(std::is_sorted(output_values.begin(), output_values.end()));
  }
}

TEST(gordeeva_t_shell_sort_batcher_merge_mpi, test_task_run) {
  boost::mpi::communicator world;
  const int size = 3000000;

  std::vector<int> input_values;
  std::vector<int> output_values(size, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_values = rand_vec(size, 0, 1000);
  }

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_values.data()));
    taskDataPar->inputs_count.emplace_back(static_cast<unsigned int>(input_values.size()));

    output_values.resize(input_values.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_values.data()));
    taskDataPar->outputs_count.emplace_back(static_cast<unsigned int>(output_values.size()));
  }

  auto testMpiTaskParallel =
      std::make_shared<gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel>(taskDataPar);

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

    ASSERT_TRUE(std::is_sorted(output_values.begin(), output_values.end()));
  }
}
