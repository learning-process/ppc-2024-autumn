// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/muradov_m_count_alpha_chars/include/ops_mpi.hpp"

std::string generate_string(size_t length) {
  std::string characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()";
  std::string result;
  result.reserve(length);

  std::default_random_engine generator;
  std::uniform_int_distribution<size_t> distribution(0, characters.size() - 1);

  for (size_t i = 0; i < length; ++i) {
    result += characters[distribution(generator)];
  }

  return result;
}

TEST(muradov_m_count_alpha_chars_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::string global_str;
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int expected_alpha_count;
  if (world.rank() == 0) {
    global_str = generate_string(9999999);

    expected_alpha_count = std::count_if(global_str.begin(), global_str.end(),
                                         [](char c) { return std::isalpha(static_cast<unsigned char>(c)); });

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  auto testMpiTaskParallel = std::make_shared<muradov_m_count_alpha_chars_mpi::AlphaCharCountTaskParallel>(taskDataPar);
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
    ASSERT_EQ(expected_alpha_count, global_count[0]);
  }
}

TEST(muradov_m_count_alpha_chars_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::string global_str;
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int expected_alpha_count;
  if (world.rank() == 0) {
    global_str = generate_string(9999999);

    expected_alpha_count = std::count_if(global_str.begin(), global_str.end(),
                                         [](char c) { return std::isalpha(static_cast<unsigned char>(c)); });

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  auto testMpiTaskParallel = std::make_shared<muradov_m_count_alpha_chars_mpi::AlphaCharCountTaskParallel>(taskDataPar);
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
    ASSERT_EQ(expected_alpha_count, global_count[0]);
  }
}