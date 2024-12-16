// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/beresnev_a_increase_contrast/include/ops_seq.hpp"

TEST(beresnev_a_increase_contrast_seq, test_pipeline_run) {
  double factor = 1.0;
  int width = 1000;
  int height = 5000;
  int max_color = 255;

  size_t file_size;

  std::vector<uint8_t> input_buffer;
  std::vector<uint8_t> inp(width * height * 3, 0);

  std::string head =
      "P6\n" + std::to_string(width) + " " + std::to_string(height) + '\n' + std::to_string(max_color) + '\n';

  file_size = head.size() + inp.size();

  input_buffer.reserve(file_size);
  input_buffer.insert(input_buffer.end(), head.begin(), head.end());
  input_buffer.insert(input_buffer.end(), inp.data(), inp.data() + inp.size());

  std::vector<uint8_t> ans_buffer;
  std::vector<uint8_t> ans(width * height * 3, 0);

  ans_buffer.reserve(file_size);
  ans_buffer.insert(ans_buffer.end(), head.begin(), head.end());
  ans_buffer.insert(ans_buffer.end(), ans.data(), ans.data() + ans.size());

  std::vector<uint8_t> out_buffer(file_size);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_buffer.data()));
  taskDataSeq->inputs_count.emplace_back(file_size);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&factor));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_buffer));
  taskDataSeq->outputs_count.emplace_back(file_size);

  auto testTaskSequential = std::make_shared<beresnev_a_increase_contrast_seq::TestTaskSequential>(taskDataSeq);
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
  ASSERT_EQ(ans_buffer, out_buffer) << ans_buffer.size() << " " << out_buffer.size();
}

TEST(beresnev_a_increase_contrast_seq, test_task_run) {
  double factor = 1.0;
  int width = 5000;
  int height = 1000;
  int max_color = 255;

  size_t file_size;

  std::vector<uint8_t> input_buffer;
  std::vector<uint8_t> inp(width * height * 3, 0);

  std::string head =
      "P6\n" + std::to_string(width) + " " + std::to_string(height) + '\n' + std::to_string(max_color) + '\n';

  file_size = head.size() + inp.size();

  input_buffer.reserve(file_size);
  input_buffer.insert(input_buffer.end(), head.begin(), head.end());
  input_buffer.insert(input_buffer.end(), inp.data(), inp.data() + inp.size());

  std::vector<uint8_t> ans_buffer;
  std::vector<uint8_t> ans(width * height * 3, 0);

  ans_buffer.reserve(file_size);
  ans_buffer.insert(ans_buffer.end(), head.begin(), head.end());
  ans_buffer.insert(ans_buffer.end(), ans.data(), ans.data() + ans.size());

  std::vector<uint8_t> out_buffer(file_size);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_buffer.data()));
  taskDataSeq->inputs_count.emplace_back(file_size);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&factor));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_buffer));
  taskDataSeq->outputs_count.emplace_back(file_size);

  auto testTaskSequential = std::make_shared<beresnev_a_increase_contrast_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_EQ(ans_buffer, out_buffer) << ans_buffer.size() << " " << out_buffer.size();
}