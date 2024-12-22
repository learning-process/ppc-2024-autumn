// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/poroshin_v_cons_conv_hull_for_bin_image_comp/include/ops_mpi.hpp"

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  const int m = 1000;
  const int n = 3000;
  std::vector<int> tmp(m * n, 1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::vector<std::pair<int, int>> result(m * n + 2);
  auto TestMPITaskParallel =
      std::make_shared<poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel>(taskDataPar);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  }

  ASSERT_TRUE(TestMPITaskParallel->validation());
  TestMPITaskParallel->pre_processing();
  TestMPITaskParallel->run();
  TestMPITaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TestMPITaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  if (world.rank() == 0) {
    std::vector<std::pair<int, int>> ans = {{0, 0}, {999, 0}, {999, 2999}, {0, 2999}, {0, 0}, {-1, -1}};
    std::vector<std::pair<int, int>> res(ans.size());
    std::copy(result.begin(), result.begin() + ans.size(), res.begin());
    ASSERT_EQ(ans, res);
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, test_task_run) {
  boost::mpi::communicator world;
  const int m = 1000;
  const int n = 3000;
  std::vector<int> tmp(m * n, 1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::vector<std::pair<int, int>> result(m * n + 2);
  auto TestMPITaskParallel =
      std::make_shared<poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel>(taskDataPar);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  }

  ASSERT_TRUE(TestMPITaskParallel->validation());
  TestMPITaskParallel->pre_processing();
  TestMPITaskParallel->run();
  TestMPITaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TestMPITaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  if (world.rank() == 0) {
    std::vector<std::pair<int, int>> ans = {{0, 0}, {999, 0}, {999, 2999}, {0, 2999}, {0, 0}, {-1, -1}};
    std::vector<std::pair<int, int>> res(ans.size());
    std::copy(result.begin(), result.begin() + ans.size(), res.begin());
    ASSERT_EQ(ans, res);
  }
}