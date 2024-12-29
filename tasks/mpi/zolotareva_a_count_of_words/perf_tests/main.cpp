// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/zolotareva_a_count_of_words/include/ops_mpi.hpp"

std::string getRandomString(int size, size_t *count_words) {
  const std::string ABC = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  std::string res;
  *count_words = 1;
  for (int i = 0; i < size; i++) {
    int x = rand() % (ABC.size() + 1);
    res += ABC[x];
    if (rand() % 2 == 1 && i < size - 1) {
      res += " ";
      (*count_words) += 1;
    }
  }
  return res;
}

TEST(mpi_zolotareva_a_count_of_words_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  std::string global_string;
  size_t answer = 0;
  size_t global_count = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_string = getRandomString(1000, &answer);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_string.data()));
    taskDataPar->inputs_count.emplace_back(global_string.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&global_count));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel = std::make_shared<zolotareva_a_count_of_words_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(answer, global_count);
  }
}

TEST(mpi_zolotareva_a_count_of_words_perf_test, test_task_run) {
  boost::mpi::communicator world;
  std::string global_string;
  size_t answer = 0;
  size_t global_count = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_string = getRandomString(1000, &answer);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_string.data()));
    taskDataPar->inputs_count.emplace_back(global_string.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&global_count));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel = std::make_shared<zolotareva_a_count_of_words_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(answer, global_count);
  }
}