// Copyright 2024 Chastov Vyacheslav
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/chastov_v_count_words_in_line/include/ops_mpi.hpp"
#include "mpi/chastov_v_count_words_in_line/src/ops_mpi.cpp"

std::vector<char> wordCountInput = createTestInput(2000);

TEST(chastov_v_count_words_in_line_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<char> input = wordCountInput;
  std::vector<int> wordsFound(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input.data())));
    taskData->inputs_count.emplace_back(input.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(wordsFound.data()));
    taskData->outputs_count.emplace_back(wordsFound.size());
  }

  auto testMpiTaskParallel = std::make_shared<chastov_v_count_words_in_line_mpi::TestMPITaskParallel>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(wordsFound[0], 30000);
  }
}

TEST(chastov_v_count_words_in_line_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<char> input = wordCountInput;
  std::vector<int> wordsFound(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input.data())));
    taskData->inputs_count.emplace_back(input.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(wordsFound.data()));
    taskData->outputs_count.emplace_back(wordsFound.size());
  }

  auto testTask = std::make_shared<chastov_v_count_words_in_line_mpi::TestMPITaskParallel>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(wordsFound[0], 30000);
  }
}