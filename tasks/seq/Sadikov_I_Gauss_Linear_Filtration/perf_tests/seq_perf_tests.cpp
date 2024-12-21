#include <gtest/gtest.h>
#include <seq/Sadikov_I_Gauss_Linear_Filtration/include/seq_task.h>

#include <iostream>
#include <thread>

#include "core/perf/include/perf.hpp"

TEST(Sadikov_I_Gauss_Linear_Filtration, image_test_pipeline_run) {
  int rows_count = 1000;
  int columns_count = 1000;
  std::vector<Point<double>> in(1000000, Point(100.0, 35.0, 78.0));
  std::vector<int> in_index{rows_count, columns_count};
  std::vector<Point<double>> out(columns_count * rows_count);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in_index[0]);
  taskData->inputs_count.emplace_back(in_index[1]);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  Sadikov_I_Gauss_Linear_Filtration::LinearFiltration filtration(taskData);
  auto testTaskSequential = std::make_shared<Sadikov_I_Gauss_Linear_Filtration::LinearFiltration>(taskData);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(Sadikov_I_Gauss_Linear_Filtration, image_test_task_run) {
  int rows_count = 1000;
  int columns_count = 1000;
  std::vector<Point<double>> in(1000000, Point(100.0, 35.0, 78.0));
  std::vector<int> in_index{rows_count, columns_count};
  std::vector<Point<double>> out(columns_count * rows_count);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in_index[0]);
  taskData->inputs_count.emplace_back(in_index[1]);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  Sadikov_I_Gauss_Linear_Filtration::LinearFiltration filtration(taskData);
  auto testTaskSequential = std::make_shared<Sadikov_I_Gauss_Linear_Filtration::LinearFiltration>(taskData);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}