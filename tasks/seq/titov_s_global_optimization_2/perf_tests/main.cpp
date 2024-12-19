// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/titov_s_global_optimization_2/include/ops_seq.hpp"

TEST(titov_s_global_optimization_2_seq, test_pipeline_run) {
  std::function<double(const titov_s_global_optimization_2_seq::Point&)> func =
      [](const titov_s_global_optimization_2_seq::Point& p) {
        return 10.0 * (p.x - 3.5) * (p.x - 3.5) + 20.0 * (p.y - 4.0) * (p.y - 4.0);
      };

  auto constraint1 = [](const titov_s_global_optimization_2_seq::Point& p) { return 6.0 - (p.x + p.y); };
  auto constraint2 = [](const titov_s_global_optimization_2_seq::Point& p) { return (2.0 * p.x + p.y) - 6; };
  auto constraint3 = [](const titov_s_global_optimization_2_seq::Point& p) { return 1.0 - (p.x - p.y); };
  auto constraint4 = [](const titov_s_global_optimization_2_seq::Point& p) { return (0.5 * p.x - p.y) + 4; };

  auto constraints_ptr =
      std::make_shared<std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>>(
          std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>{
              constraint1, constraint2, constraint3, constraint4});

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
  taskDataSeq->inputs_count.emplace_back(constraints_ptr->size());

  std::vector<titov_s_global_optimization_2_seq::Point> out(1, {0.0, 0.0});
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto optimizationTask = std::make_shared<titov_s_global_optimization_2_seq::GlobalOpt2Sequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(optimizationTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_NEAR(out[0].x, 2.5, 0.1);
  ASSERT_NEAR(out[0].y, 3.5, 0.1);
}

TEST(titov_s_global_optimization_2_seq, test_task_run) {
  std::function<double(const titov_s_global_optimization_2_seq::Point&)> func =
      [](const titov_s_global_optimization_2_seq::Point& p) {
        return 10.0 * (p.x - 3.5) * (p.x - 3.5) + 20.0 * (p.y - 4.0) * (p.y - 4.0);
      };

  auto constraint1 = [](const titov_s_global_optimization_2_seq::Point& p) { return 6.0 - (p.x + p.y); };
  auto constraint2 = [](const titov_s_global_optimization_2_seq::Point& p) { return (2.0 * p.x + p.y) - 6; };
  auto constraint3 = [](const titov_s_global_optimization_2_seq::Point& p) { return 1.0 - (p.x - p.y); };
  auto constraint4 = [](const titov_s_global_optimization_2_seq::Point& p) { return (0.5 * p.x - p.y) + 4; };

  auto constraints_ptr =
      std::make_shared<std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>>(
          std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>{
              constraint1, constraint2, constraint3, constraint4});

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
  taskDataSeq->inputs_count.emplace_back(constraints_ptr->size());

  std::vector<titov_s_global_optimization_2_seq::Point> out(1, {0.0, 0.0});
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto optimizationTask = std::make_shared<titov_s_global_optimization_2_seq::GlobalOpt2Sequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(optimizationTask);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_NEAR(out[0].x, 2.5, 0.1);
  ASSERT_NEAR(out[0].y, 3.5, 0.1);
}
