#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/sozonov_i_gaussian_method_horizontal_strip_scheme/include/ops_seq.hpp"

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_seq, test_pipeline_run) {
  const int count = 100;

  // Create data
  std::vector<double> in(count * (count + 1));
  std::vector<double> out(count, 0);
  std::vector<double> ans(count);

  for (size_t i = 0; i < count; ++i) {
    in[i * (count + 1) + i] = 1;
    in[i * (count + 1) + count] = i + 1;
  }
  std::iota(ans.begin(), ans.end(), 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(count);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<sozonov_i_gaussian_method_horizontal_strip_scheme_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_EQ(ans, out);
}

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_seq, test_task_run) {
  const int count = 100;

  // Create data
  std::vector<double> in(count * (count + 1));
  std::vector<double> out(count, 0);
  std::vector<double> ans(count);

  for (size_t i = 0; i < count; ++i) {
    in[i * (count + 1) + i] = 1;
    in[i * (count + 1) + count] = i + 1;
  }
  std::iota(ans.begin(), ans.end(), 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(count);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<sozonov_i_gaussian_method_horizontal_strip_scheme_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_EQ(ans, out);
}
