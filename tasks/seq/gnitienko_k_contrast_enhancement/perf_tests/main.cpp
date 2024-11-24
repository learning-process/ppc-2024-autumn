#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/gnitienko_k_contrast_enhancement/include/ops_seq.hpp"

namespace gnitienko_k_functions {

std::vector<int> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 255;
  }
  return vec;
}
}  // namespace gnitienko_k_functions

TEST(gnitienko_k_contrast_enhancement_seq, test_pipeline_run) {
  const int size = 5000000;

  // Create data
  std::vector<int> in = gnitienko_k_functions::getRandomVector(size);
  std::vector<int> out(size, 0);
  double contrast_factor = 3.0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&contrast_factor));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  std::vector<int> expected_out(size, 0);
  for (size_t i = 0; i < size; i++)
    expected_out[i] = std::clamp(static_cast<int>((in[i] - 128) * contrast_factor + 128), 0, 255);

  // Create Task
  auto testTaskSequential = std::make_shared<gnitienko_k_contrast_enhancement_seq::ContrastEnhanceSeq>(taskDataSeq);

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
  ASSERT_EQ(out, expected_out);
}

TEST(gnitienko_k_contrast_enhancement_seq, test_task_run) {
  const int size = 5000000;

  // Create data
  std::vector<int> in = gnitienko_k_functions::getRandomVector(size);
  std::vector<int> out(size, 0);
  double contrast_factor = 3.0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&contrast_factor));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  std::vector<int> expected_out(size, 0);
  for (size_t i = 0; i < size; i++)
    expected_out[i] = std::clamp(static_cast<int>((in[i] - 128) * contrast_factor + 128), 0, 255);

  // Create Task
  auto testTaskSequential = std::make_shared<gnitienko_k_contrast_enhancement_seq::ContrastEnhanceSeq>(taskDataSeq);

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
  ASSERT_EQ(out, expected_out);
}
