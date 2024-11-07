// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/zolotareva_a_count_of_words/include/ops_mpi.hpp"

TEST(mpi_zolotareva_a_count_of_words_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  std::string global_string;
  size_t global_count = 0;
  size_t answer = 162;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_string =
        "I was too young to be other than awed and puzzled by Doc Marlowe when I knew him. I was only sixteen when he "
        "died. He was sixty-seven. There was that vast difference in our ages and there was a vaster difference in our "
        "backgrounds. Doc Marlowe was a medicine-show man. He had been a lot of other things, too: a circus man, the "
        "proprietor of a concession at Coney Island, a saloon-keeper; but in his fifties he had travelled around with "
        "a tent-show troupe made up of a Mexican named Chickalilli, who threw knives, and a man called Professor "
        "Jones, who played the banjo. Doc Marlowe would come out after the entertainment and harangue the crowd and "
        "sell bottles of medicine for all kinds of ailments. I found out all this about him gradually, toward the "
        "last, and after he died. When I first knew him, he represented the Wild West to me, and there was nobody I "
        "admired so much.";
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_string.data()));
    taskDataPar->inputs_count.emplace_back(global_string.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_count));
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
  size_t global_count = 0;
  size_t answer = 162;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_string =
        "I was too young to be other than awed and puzzled by Doc Marlowe when I knew him. I was only sixteen when he "
        "died. He was sixty-seven. There was that vast difference in our ages and there was a vaster difference in our "
        "backgrounds. Doc Marlowe was a medicine-show man. He had been a lot of other things, too: a circus man, the "
        "proprietor of a concession at Coney Island, a saloon-keeper; but in his fifties he had travelled around with "
        "a tent-show troupe made up of a Mexican named Chickalilli, who threw knives, and a man called Professor "
        "Jones, who played the banjo. Doc Marlowe would come out after the entertainment and harangue the crowd and "
        "sell bottles of medicine for all kinds of ailments. I found out all this about him gradually, toward the "
        "last, and after he died. When I first knew him, he represented the Wild West to me, and there was nobody I "
        "admired so much.";
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_string.data()));
    taskDataPar->inputs_count.emplace_back(global_string.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_count));
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