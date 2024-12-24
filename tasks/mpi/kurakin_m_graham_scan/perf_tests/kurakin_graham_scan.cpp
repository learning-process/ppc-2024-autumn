// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kurakin_m_graham_scan/include/kurakin_graham_scan_ops_mpi.hpp"

TEST(kurakin_m_graham_scan_mpi_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;

  int count_point = 100000;
  std::vector<double> points;

  int scan_size_par;
  std::vector<double> scan_points_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    points = std::vector<double>(count_point * 2);

    points[0] = count_point / 2;
    points[1] = (-1) * count_point / 2;

    for (int i = 2; i < count_point * 2; i += 2) {
      points[i] = count_point / 2 - i / 2;
      points[i + 1] = count_point / 2 - i / 2;
    }

    scan_points_par = std::vector<double>(count_point * 2);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataPar->inputs_count.emplace_back(points.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_points_par.size());
  }

  auto testMpiTaskParallel = std::make_shared<kurakin_m_graham_scan_mpi::TestMPITaskParallel>(taskDataPar);
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
    int ans_size = count_point;
    std::vector<double> ans(ans_size * 2);

    ans[0] = ans_size / 2;
    ans[1] = (-1) * ans_size / 2;

    for (int i = 2; i < ans_size * 2; i += 2) {
      ans[i] = ans_size / 2 - i / 2;
      ans[i + 1] = ans_size / 2 - i / 2;
    }

    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(ans_size, scan_size_par);
    for (int i = 0; i < ans_size * 2; i += 2) {
      ASSERT_EQ(ans[i], scan_points_par[i]);
      ASSERT_EQ(ans[i + 1], scan_points_par[i + 1]);
    }
  }
}

TEST(kurakin_m_graham_scan_mpi_perf_test, test_task_run) {
  boost::mpi::communicator world;

  int count_point = 100000;
  std::vector<double> points;

  int scan_size_par;
  std::vector<double> scan_points_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    points = std::vector<double>(count_point * 2);

    points[0] = count_point / 2;
    points[1] = (-1) * count_point / 2;

    for (int i = 2; i < count_point * 2; i += 2) {
      points[i] = count_point / 2 - i / 2;
      points[i + 1] = count_point / 2 - i / 2;
    }

    scan_points_par = std::vector<double>(count_point * 2);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataPar->inputs_count.emplace_back(points.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_points_par.size());
  }

  auto testMpiTaskParallel = std::make_shared<kurakin_m_graham_scan_mpi::TestMPITaskParallel>(taskDataPar);
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
    int ans_size = count_point;
    std::vector<double> ans(ans_size * 2);

    ans[0] = ans_size / 2;
    ans[1] = (-1) * ans_size / 2;

    for (int i = 2; i < ans_size * 2; i += 2) {
      ans[i] = ans_size / 2 - i / 2;
      ans[i + 1] = ans_size / 2 - i / 2;
    }

    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(ans_size, scan_size_par);
    for (int i = 0; i < ans_size; i += 2) {
      ASSERT_EQ(ans[i], scan_points_par[i]);
      ASSERT_EQ(ans[i + 1], scan_points_par[i + 1]);
    }
  }
}
