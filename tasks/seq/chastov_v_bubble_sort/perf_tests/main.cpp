#include <gtest/gtest.h>

#include <random>

#include "core/perf/include/perf.hpp"
#include "seq/chastov_v_bubble_sort/include/ops_seq.hpp"

TEST(chastov_v_bubble_sort, test_pipeline_run) {
  const size_t count = 100;

  // Create data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 1000);
  const int num = dis(gen);
  std::vector<int> in(count, num);
  std::vector<int> out(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<chastov_v_bubble_sort::TestTaskSequential<int>>(taskDataSeq);

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
  int *tmp = reinterpret_cast<int *>(out.data());
  int error_count = 0;
  for (size_t i = 0; i < count; i++) {
    if (tmp[i] != in[i]) error_count++;
  }
  ASSERT_EQ(error_count, 0);
}

TEST(chastov_v_bubble_sort, test_task_run) {
  const size_t count = 1000;

  // Create data
  std::vector<double> in(count);
  std::vector<double> out(count);
  std::vector<double> etalon(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  auto max = static_cast<double>(1000000);
  auto min = static_cast<double>(-1000000);
  std::srand(std::time(nullptr));
  for (size_t i = 0; i < count; i++) etalon[i] = in[i] = min + static_cast<double>(rand()) / RAND_MAX * (max - min);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<chastov_v_bubble_sort::TestTaskSequential<double>>(taskDataSeq);

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
  std::sort(etalon.begin(), etalon.end(), [](double a, double b) { return a < b; });
  auto *tmp = reinterpret_cast<double *>(out.data());
  int error_count = 0;
  for (size_t i = 0; i < count; i++) {
    if (tmp[i] != etalon[i]) error_count++;
  }
  ASSERT_EQ(error_count, 0);
}