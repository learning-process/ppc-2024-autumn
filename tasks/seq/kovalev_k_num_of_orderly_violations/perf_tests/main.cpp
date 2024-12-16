#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kovalev_k_num_of_orderly_violations/include/header.hpp"

TEST(kovalev_k_num_of_orderly_violations_seq, test_pipeline_run) {
  const size_t length = 10;
  const int alpha = 1;
  std::vector<int> in(length, alpha);
  std::vector<size_t> out(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskSeq->inputs_count.emplace_back(in.size());
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  auto testTaskSequential =
      std::make_shared<kovalev_k_num_of_orderly_violations_seq::NumOfOrderlyViolations<int>>(taskSeq);
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
  size_t result = 0;
  ASSERT_EQ(result, out[0]);
}

TEST(kovalev_k_num_of_orderly_violations_mpi, test_task_run) {
  std::vector<double> g_vec;
  std::vector<size_t> g_num_viol(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  size_t length = 1000000;
  double max = 1000000;
  double min = -1000000;
  g_vec = std::vector<double>(length);
  std::srand(std::time(nullptr));
  for (size_t i = 0; i < length; i++) g_vec[i] = min + static_cast<double>(rand()) / RAND_MAX * (max - min);
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(g_vec.data()));
  taskSeq->inputs_count.emplace_back(g_vec.size());
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(g_num_viol.data()));
  taskSeq->outputs_count.emplace_back(g_num_viol.size());
  auto testTaskSequential =
      std::make_shared<kovalev_k_num_of_orderly_violations_seq::NumOfOrderlyViolations<double>>(taskSeq);
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
  size_t res = 0;
  for (size_t i = 1; i < length; i++)
    if (g_vec[i - 1] > g_vec[i]) res++;
  ASSERT_EQ(res, g_num_viol[0]);
}