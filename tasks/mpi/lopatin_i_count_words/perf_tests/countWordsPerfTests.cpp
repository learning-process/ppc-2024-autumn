#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/lopatin_i_count_words/include/countWordsMPIHeader.hpp"

std::vector<char> testData = lopatin_i_count_words_mpi::generateLongString(2000);

TEST(lopatin_i_count_words_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<char> input = testData;
  std::vector<int> wordCount(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input.data())));
    taskData->inputs_count.emplace_back(input.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(wordCount.data()));
    taskData->outputs_count.emplace_back(wordCount.size());
  }

  auto testTask = std::make_shared<lopatin_i_count_words_mpi::TestMPITaskParallel>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(wordCount[0], 30000);
  }
}

TEST(lopatin_i_count_words_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<char> input = testData;
  std::vector<int> wordCount(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input.data())));
    taskData->inputs_count.emplace_back(input.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(wordCount.data()));
    taskData->outputs_count.emplace_back(wordCount.size());
  }

  auto testTask = std::make_shared<lopatin_i_count_words_mpi::TestMPITaskParallel>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(wordCount[0], 30000);
  }
}