#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/vedernikova_k_monte_carlo_integration_seq/include/ops_seq.hpp"

TEST(vedernikova_k_monte_carlo_integration_seq, test_pipeline_run) {
  double ax = -1.0;
  double bx = 1.0;
  double ay = -2.0;
  double by = 2.0;
  size_t num_point = 500000;

  double out = 0.0;
  double expected_res = 8.0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ax));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bx));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ay));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&by));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_point));

  taskDataSeq->inputs_count.emplace_back(taskDataSeq->inputs.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);
  // Create Task
  auto testTaskSequential =
      std::make_shared<vedernikova_k_monte_carlo_integration_seq::TestTaskSequential>(taskDataSeq);

  testTaskSequential->f = [](double x, double y) { return 1 - (1 / 3) * x - 0.25 * y; };
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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  EXPECT_NEAR(expected_res, out, 1.0);
}

TEST(vedernikova_k_monte_carlo_integration_seq, test_task_run) {
  double ax = -1.0;
  double bx = 1.0;
  double ay = -2.0;
  double by = 2.0;
  size_t num_point = 500000;

  double out = 0.0;
  double expected_res = 8.0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ax));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bx));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ay));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&by));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_point));

  taskDataSeq->inputs_count.emplace_back(taskDataSeq->inputs.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);
  // Create Task
  auto testTaskSequential =
      std::make_shared<vedernikova_k_monte_carlo_integration_seq::TestTaskSequential>(taskDataSeq);

  testTaskSequential->f = [](double x, double y) { return 1 - (1 / 3) * x - 0.25 * y; };
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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  EXPECT_NEAR(expected_res, out, 1.0);
}
