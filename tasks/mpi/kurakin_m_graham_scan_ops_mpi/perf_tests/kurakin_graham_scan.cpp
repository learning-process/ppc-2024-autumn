// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kurakin_m_graham_scan_ops_mpi/include/kurakin_graham_scan_ops_mpi.hpp"

TEST(kurakin_m_graham_scan_mpi_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;

  int count_point = 1000;
  std::vector<double> point_x;
  std::vector<double> point_y;

  int scan_size_par;
  std::vector<double> scan_x_par;
  std::vector<double> scan_y_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    point_x = std::vector<double>(count_point);
    point_y = std::vector<double>(count_point);

    point_x[0] = count_point / 2;
    point_y[0] = (-1) * count_point / 2;

    for (int i = 1; i < count_point; i++) {
      point_x[i] = count_point / 2 - i;
      point_y[i] = count_point / 2 - i;
    }

    scan_x_par = std::vector<double>(count_point);
    scan_y_par = std::vector<double>(count_point);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
    taskDataPar->inputs_count.emplace_back(point_x.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
    taskDataPar->inputs_count.emplace_back(point_y.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_x_par.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_y_par.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<kurakin_m_graham_scan_mpi::TestMPITaskParallel>(taskDataPar);
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
    std::vector<double> ans_x(ans_size);
    std::vector<double> ans_y(ans_size);

    ans_x[0] = ans_size / 2;
    ans_y[0] = (-1) * ans_size / 2;

    for (int i = 1; i < ans_size; i++) {
      ans_x[i] = ans_size / 2 - i;
      ans_y[i] = ans_size / 2 - i;
    }

    ppc::core::Perf::print_perf_statistic(perfResults);
    for (int i = 0; i < ans_size; i++) {
      EXPECT_EQ(ans_x[i], scan_x_par[i]);
      EXPECT_EQ(ans_y[i], scan_y_par[i]);
    }
  }
}

TEST(kurakin_m_graham_scan_mpi_perf_test, test_task_run) {
  boost::mpi::communicator world;

  int count_point = 1000;
  std::vector<double> point_x;
  std::vector<double> point_y;

  int scan_size_par;
  std::vector<double> scan_x_par;
  std::vector<double> scan_y_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    point_x = std::vector<double>(count_point);
    point_y = std::vector<double>(count_point);

    point_x[0] = count_point / 2;
    point_y[0] = (-1) * count_point / 2;

    for (int i = 1; i < count_point; i++) {
      point_x[i] = count_point / 2 - i;
      point_y[i] = count_point / 2 - i;
    }

    scan_x_par = std::vector<double>(count_point);
    scan_y_par = std::vector<double>(count_point);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
    taskDataPar->inputs_count.emplace_back(point_x.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
    taskDataPar->inputs_count.emplace_back(point_y.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_x_par.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_y_par.size());
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
    std::vector<double> ans_x(ans_size);
    std::vector<double> ans_y(ans_size);

    ans_x[0] = ans_size / 2;
    ans_y[0] = (-1) * ans_size / 2;

    for (int i = 1; i < ans_size; i++) {
      ans_x[i] = ans_size / 2 - i;
      ans_y[i] = ans_size / 2 - i;
    }

    ppc::core::Perf::print_perf_statistic(perfResults);
    for (int i = 0; i < ans_size; i++) {
      EXPECT_EQ(ans_x[i], scan_x_par[i]);
      EXPECT_EQ(ans_y[i], scan_y_par[i]);
    }
  }
}
