
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/vasenkov_a_char_freq/include/ops_mpi.hpp"

TEST(vasenkov_a_char_freq_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<char> global_str;
  std::vector<int32_t> global_count(1, 0);
  char target_char = 'a';

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_str;
  if (world.rank() == 0) {
    count_size_str = 120;
    global_str = std::vector<char>(count_size_str, 'a');
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&target_char));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  auto testMpiTaskParallel = std::make_shared<vasenkov_a_char_freq_mpi::CharFrequencyParallel>(taskDataPar);
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
    ASSERT_EQ(count_size_str, global_count[0]);
  }
}

TEST(vasenkov_a_char_freq_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<char> global_str;
  std::vector<int32_t> global_count(1, 0);
  char target_char = 'a';

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_str;
  if (world.rank() == 0) {
    count_size_str = 120;
    global_str = std::vector<char>(count_size_str, 'a');
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&target_char));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  auto testMpiTaskParallel = std::make_shared<vasenkov_a_char_freq_mpi::CharFrequencyParallel>(taskDataPar);
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
    ASSERT_EQ(count_size_str, global_count[0]);
  }
}
