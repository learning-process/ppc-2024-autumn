#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <string>

#include "core/perf/include/perf.hpp"
#include "mpi/lopatin_i_count_words/include/countWordsMPIHeader.hpp"

TEST(word_count_mpi, test_pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  std::string input = "This is a long sentence for performance testing of the word count algorithm using MPI";
  std::vector<int> word_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input.c_str())));
    taskData->inputs_count.emplace_back(input.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(word_count.data()));
    taskData->outputs_count.emplace_back(word_count.size());
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
    ASSERT_EQ(word_count[0], 14);
  }
}

TEST(word_count_mpi, test_task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  std::string input = "This is another long sentence for performance testing of the word count algorithm using MPI";
  std::vector<int> word_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input.c_str())));
    taskData->inputs_count.emplace_back(input.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(word_count.data()));
    taskData->outputs_count.emplace_back(word_count.size());
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
    ASSERT_EQ(word_count[0], 15);
  }
}
