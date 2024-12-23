#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/zolotareva_a_SLE_gradient_method/include/ops_seq.hpp"

TEST(sequential_zolotareva_a_SLE_gradient_method_perf_test, test_pipeline_run) {
  const uint32_t n = 1000;
  std::vector<double> A(n * n, 0);
  std::vector<double> b(n, 0);
  std::vector<double> x(n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
  taskDataSeq->inputs_count.push_back(n * n);
  taskDataSeq->inputs_count.push_back(n);
  taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
  taskDataSeq->outputs_count.push_back(x.size());

  auto testTaskSequential = std::make_shared<zolotareva_a_SLE_gradient_method_seq::TestTaskSequential>(taskDataSeq);

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

  ASSERT_EQ(n, taskDataSeq->inputs_count[1]);
}

TEST(sequential_zolotareva_a_SLE_gradient_method_perf_test, test_task_run) {
  const uint32_t n = 1000;
  std::vector<double> A(n * n, 0);
  std::vector<double> b(n, 0);
  std::vector<double> x(n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
  taskDataSeq->inputs_count.push_back(n * n);
  taskDataSeq->inputs_count.push_back(n);
  taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
  taskDataSeq->outputs_count.push_back(x.size());

  auto testTaskSequential = std::make_shared<zolotareva_a_SLE_gradient_method_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_EQ(n, taskDataSeq->inputs_count[1]);
}