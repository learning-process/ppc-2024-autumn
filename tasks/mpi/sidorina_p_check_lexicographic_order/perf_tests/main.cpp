// Copyright 2024 Nesterov Alexander
#include <gtest/gtest.h>
#include <boost/mpi/timer.hpp>
#include <vector>
#include "core/perf/include/perf.hpp"
#include "mpi/sidorina_p_check_lexicographic_order/include/ops_mpi.hpp"

TEST(sidorina_p_check_lexicographic_order_mpi, Test_0) {
  boost::mpi::communicator world;
  std::vector<char> str1(40000000, 'e');
  std::vector<char> str2(39999999, 'e');
  str2.push_back('f');
  std::vector<std::vector<char>> str_ = {str1, str2};
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[1].data()));
    taskDataPar->inputs_count.emplace_back(str_.size());
    taskDataPar->inputs_count.emplace_back(str_[0].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<sidorina_p_check_lexicographic_order_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(res[0], 0);
  }
}
TEST(sidorina_p_check_lexicographic_order_mpi, Test_1) {
  boost::mpi::communicator world;
  std::vector<char> str1(40000000, 'f');
  std::vector<char> str2(39999999, 'f');
  str2.push_back('a');
  std::vector<std::vector<char>> str_ = {str1, str2};
  std::vector<int32_t> res(1, 0);
  
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[1].data()));
    taskDataPar->inputs_count.emplace_back(str_.size());
    taskDataPar->inputs_count.emplace_back(str_[0].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<sidorina_p_check_lexicographic_order_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(res[0], 1);
  }
}