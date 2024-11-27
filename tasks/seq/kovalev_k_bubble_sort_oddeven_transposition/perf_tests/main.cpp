#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/kovalev_k_bubble_sort_oddeven_transposition/include/header.hpp"

TEST(kovalev_k_bubble_sort_oddeven_transposition_seq, test_pipeline_run) {
  const size_t length = 10;
  std::srand(std::time(nullptr));
  const int alpha = rand();
  std::vector<int> in(length, alpha);
  std::vector<int> out(length);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskSeq->inputs_count.emplace_back(in.size());
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  auto testTaskSequential =
      std::make_shared<kovalev_k_bubble_sort_oddeven_transposition_seq::BubbleSortOddEvenTransposition<int>>(taskSeq);
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
  int *tmp = reinterpret_cast<int *>(out.data());
  int count_viol = 0;
  for (size_t i = 0; i < length; i++) {
    if (tmp[i] != in[i]) count_viol++;
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_bubble_sort_oddeven_transposition_seq, test_task_run) {
  const size_t length = 10000;
  std::vector<double> in(length);
  std::vector<double> out(length);
  std::vector<double> etalon(length);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  auto max = static_cast<double>(1000000);
  auto min = static_cast<double>(-1000000);
  std::srand(std::time(nullptr));
  for (size_t i = 0; i < length; i++) etalon[i] = in[i] = min + static_cast<double>(rand()) / RAND_MAX * (max - min);
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskSeq->inputs_count.emplace_back(in.size());
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  auto testTaskSequential =
      std::make_shared<kovalev_k_bubble_sort_oddeven_transposition_seq::BubbleSortOddEvenTransposition<double>>(
          taskSeq);
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
  std::sort(etalon.begin(), etalon.end(), [](double a, double b) { return a < b; });
  auto *tmp = reinterpret_cast<double *>(out.data());
  int count_viol = 0;
  for (size_t i = 0; i < length; i++) {
    if (tmp[i] != etalon[i]) count_viol++;
  }
  ASSERT_EQ(count_viol, 0);
}