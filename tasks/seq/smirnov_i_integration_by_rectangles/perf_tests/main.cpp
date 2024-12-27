#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/smirnov_i_integration_by_rectangles/include/ops_seq.hpp"
double f1(double x) { return x * x; }
TEST(smirnov_i_integration_by_rectangles_seq, test_pipeline_run) {
  double left = 0;
  double right = 1;
  int n = 1000;
  double expected_result = 1. / 3;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&left));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&right));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataSeq->inputs_count.emplace_back(3);
  std::vector<double> res(1, 0.0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  taskDataSeq->outputs_count.emplace_back(1);
  auto testTaskSequential = std::make_shared<smirnov_i_integration_by_rectangles::TestMPITaskSequential>(taskDataSeq);
  testTaskSequential->set_function(f1);
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
  EXPECT_NEAR(res[0], expected_result, 1e-5);
}

TEST(smirnov_i_integration_by_rectangles_seq, test_task_run) {
  double left = 0;
  double right = 1;
  int n = 1000;
  double expected_result = 1. / 3;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&left));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&right));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataSeq->inputs_count.emplace_back(3);
  std::vector<double> res(1, 0.0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  taskDataSeq->outputs_count.emplace_back(1);
  auto testTaskSequential = std::make_shared<smirnov_i_integration_by_rectangles::TestMPITaskSequential>(taskDataSeq);
  testTaskSequential->set_function(f1);
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
  EXPECT_NEAR(res[0], expected_result, 1e-5);
}
