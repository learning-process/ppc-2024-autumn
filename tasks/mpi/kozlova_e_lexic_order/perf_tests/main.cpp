// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kozlova_e_lexic_order/include/ops_mpi.hpp"

TEST(kozlova_e_lexic_order_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  const int size = 2000000;
  std::vector<int> resMPI;
  std::vector<int> expect = {1, 1};
  std::string str1;
  std::string str2;
  str1.resize(size, '\0');
  str2.resize(size, '\0');
  for (int i = 0; i < 30; i++) {
    str1[i] = 'a';
    str2[i] = 'b';
  }
  for (int i = 30; i < 60; i++) {
    str1[i] = 'C';
    str2[i] = 'd';
  }
  for (int i = 60; i < size; i++) {
    str1[i] = 'e';
    str2[i] = 'f';
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    resMPI = {0, 0};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(str1.c_str())));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(str2.c_str())));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(2));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(static_cast<uint32_t>(2));
  }

  auto testMpiTaskParallel = std::make_shared<kozlova_e_lexic_order_mpi::StringComparatorMPI>(taskDataPar);
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
    ASSERT_EQ(resMPI, expect);
  }
}

TEST(kozlova_e_lexic_order_mpi, test_task_run) {
  boost::mpi::communicator world;
  const int size = 2000000;
  std::vector<int> resMPI;
  std::vector<int> expect = {1, 1};
  std::string str1;
  std::string str2;
  str1.resize(size, '\0');
  str2.resize(size, '\0');
  for (int i = 0; i < 30; i++) {
    str1[i] = 'a';
    str2[i] = 'b';
  }
  for (int i = 30; i < 60; i++) {
    str1[i] = 'C';
    str2[i] = 'd';
  }
  for (int i = 60; i < size; i++) {
    str1[i] = 'e';
    str2[i] = 'f';
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    
    resMPI = {0, 0};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(str1.c_str())));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(str2.c_str())));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(2));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(static_cast<uint32_t>(2));
  }

  auto testMpiTaskParallel = std::make_shared<kozlova_e_lexic_order_mpi::StringComparatorMPI>(taskDataPar);
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
    ASSERT_EQ(resMPI, expect);
  }
}
