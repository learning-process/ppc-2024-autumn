#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/burykin_m_word_count/include/ops_mpi.hpp"

std::vector<char> burykin_m_word_count::RandomSentence(int size) {
  std::vector<char> vec(size);
  std::random_device dev;
  std::mt19937 gen(dev());
  if (size > 0) {
    vec[size - 1] = 0x61 + gen() % 26;
    vec[0] = 0x41 + gen() % 26;
  }
  for (int i = 1; i < size - 1; i++) {
    if (vec[i - 1] != ' ' && gen() % 4 == 0) {
      vec[i] = ' ';
    } else {
      vec[i] = 0x61 + gen() % 26;
    }
  }
  return vec;
}

TEST(burykin_m_word_count_MPI_perf, test_pipeline_run) {
  int length = 10000;

  // Create data
  std::vector<char> input(length, 'a');
  std::vector<int> wordCount(1, 0);

  boost::mpi::communicator world;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(input.size());
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(wordCount.data()));
    taskDataPar->outputs_count.emplace_back(wordCount.size());
  }

  // Create Task
  auto testMpiTaskParallel = std::make_shared<burykin_m_word_count::TestTaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(1, wordCount[0]);
  }
}

TEST(burykin_m_word_count_MPI_perf, test_task_run) {
  int length = 10000;

  // Create data
  std::vector<char> input(length, 'a');
  std::vector<int> wordCount(1, 0);

  boost::mpi::communicator world;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(input.size());
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(wordCount.data()));
    taskDataPar->outputs_count.emplace_back(wordCount.size());
  }

  // Create Task
  auto testMpiTaskParallel = std::make_shared<burykin_m_word_count::TestTaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(1, wordCount[0]);
  }
}
