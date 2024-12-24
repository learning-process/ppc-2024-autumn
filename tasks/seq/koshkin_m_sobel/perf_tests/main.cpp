#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "../include/ops_seq.hpp"
#include "core/perf/include/perf.hpp"

static std::vector<uint8_t> make_img(size_t width, size_t height) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distrib(0, 255);
  std::vector<uint8_t> vec(width * height);
  for (size_t i = 0; i < width * height; i++) {
    vec[i] = distrib(gen);
  }
  return vec;
}

TEST(koshkin_m_sobel_seq_perf_test, test_pipeline_run) {
  const auto width = 1200;
  const auto height = 1200;
  std::vector<uint8_t> in = make_img(width, height);
  std::vector<uint8_t> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs = {reinterpret_cast<uint8_t *>(in.data())};
  taskDataSeq->inputs_count = {width, height};
  taskDataSeq->outputs = {reinterpret_cast<uint8_t *>(out.data())};
  taskDataSeq->outputs_count = {width, height};

  // Create Task
  auto testTaskSequential = std::make_shared<koshkin_m_sobel_seq::TestTaskSequential>(taskDataSeq);

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
}

TEST(koshkin_m_sobel_seq_perf_test, test_task_run) {
  const auto width = 1200;
  const auto height = 1200;
  std::vector<uint8_t> in = make_img(width, height);
  std::vector<uint8_t> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs = {reinterpret_cast<uint8_t *>(in.data())};
  taskDataSeq->inputs_count = {width, height};
  taskDataSeq->outputs = {reinterpret_cast<uint8_t *>(out.data())};
  taskDataSeq->outputs_count = {width, height};

  // Create Task
  auto testTaskSequential = std::make_shared<koshkin_m_sobel_seq::TestTaskSequential>(taskDataSeq);

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
}
