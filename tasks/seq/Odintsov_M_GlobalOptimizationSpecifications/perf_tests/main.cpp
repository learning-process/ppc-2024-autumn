
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/Odintsov_M_GlobalOptimizationSpecifications/include/ops_seq.hpp"

TEST(Odintsov_m_SequentialOptimal_perf_test, test_pipeline_run) {
  // Create data
  double step = 0.01;
  std::vector<double> area = {-10, 10, -10, 10};
  std::vector<double> func = {5, 5};
  std::vector<double> constraint = {1, 1, 1, 2, 1, 1, 2, 3, 1, 4, 1, 1};
  std::vector<double> out = {0};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));
  taskDataSeq->inputs_count.emplace_back(4);  // Количество ограничений
  taskDataSeq->inputs_count.emplace_back(0);  // Режим
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  // Create Task
  auto testClass =
      std::make_shared<Odintsov_M_GlobalOptimizationSpecifications_seq::GlobalOptimizationSpecificationsSequential>(
          taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 15;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testClass);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);

  EXPECT_NEAR(46.08, out[0], 0.000001);
}

TEST(Odintsov_m_SequentialOptimal_perf_test, test_task_run) {
  // Create data
  double step = 0.01;
  std::vector<double> area = {-10, 10, -10, 10};
  std::vector<double> func = {5, 5};
  std::vector<double> constraint = {1, 1, 1, 2, 1, 1, 2, 3, 1, 4, 1, 1};
  std::vector<double> out(1, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));
  taskDataSeq->inputs_count.emplace_back(4);  // Количество ограничений
  taskDataSeq->inputs_count.emplace_back(0);  // Режим
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testClass =
      std::make_shared<Odintsov_M_GlobalOptimizationSpecifications_seq::GlobalOptimizationSpecificationsSequential>(
          taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 15;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testClass);

  perfAnalyzer->task_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);

  EXPECT_NEAR(46.08, out[0], 0.000001);
}