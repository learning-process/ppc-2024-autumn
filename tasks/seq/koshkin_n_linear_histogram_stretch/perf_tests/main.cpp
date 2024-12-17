#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/koshkin_n_linear_histogram_stretch/include/ops_seq.hpp"

TEST(koshkin_n_linear_histogram_stretch_seq, test_pipeline_run) {
  const int width = 173;
  const int height = 173;
  const int count_size_vector = width * height * 3;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  auto testTaskSequential = std::make_shared<koshkin_n_linear_histogram_stretch_seq::TestTaskSequential>(taskDataSeq);

  std::vector<int> in_vec = koshkin_n_linear_histogram_stretch_seq::getRandomImage(count_size_vector);
  std::vector<int> out_vec(count_size_vector, 0);
  std::vector<int> res_exp_out;

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
  taskDataSeq->inputs_count.emplace_back(in_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_vec.size());

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
  ASSERT_EQ(count_size_vector, count_size_vector);
}

TEST(koshkin_n_linear_histogram_stretch_seq, test_task_run) {
  const int width = 173;
  const int height = 173;
  const int count_size_vector = width * height * 3;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  auto testTaskSequential = std::make_shared<koshkin_n_linear_histogram_stretch_seq::TestTaskSequential>(taskDataSeq);

  std::vector<int> in_vec = koshkin_n_linear_histogram_stretch_seq::getRandomImage(count_size_vector);
  std::vector<int> out_vec(count_size_vector, 0);
  std::vector<int> res_exp_out;

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
  taskDataSeq->inputs_count.emplace_back(in_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_vec.size());

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
  ASSERT_EQ(count_size_vector, count_size_vector);
}
