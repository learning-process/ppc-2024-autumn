#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/zaitsev_a_jarvis/include/ops_seq.hpp"
#include "seq/zaitsev_a_jarvis/include/point.hpp"

namespace zaitsev_a_jarvis_seq {
std::vector<zaitsev_a_jarvis_seq::Point<int>> get_random_vector(int size) {
  std::vector<zaitsev_a_jarvis_seq::Point<int>> vec(size, {0, 0});
  std::random_device dev;
  std::mt19937 gen(dev());
  for (int i = 0; i < size; i++) {
    vec[i].x = gen() % 100;
    vec[i].y = gen() % 100;
  }
  return vec;
}
}  // namespace zaitsev_a_jarvis_seq

TEST(zaitsev_a_jarvis_seq_perf, test_pipeline_run) {
  const int size = 100;

  std::vector<zaitsev_a_jarvis_seq::Point<int>> in = zaitsev_a_jarvis_seq::get_random_vector(size);

  in[0] = zaitsev_a_jarvis_seq::Point<int>{-1, -1};
  in[1] = zaitsev_a_jarvis_seq::Point<int>{-1, 101};
  in[2] = zaitsev_a_jarvis_seq::Point<int>{101, 101};
  in[3] = zaitsev_a_jarvis_seq::Point<int>{101, -1};

  std::vector<zaitsev_a_jarvis_seq::Point<int>> out(size, {0, 0});
  std::vector<zaitsev_a_jarvis_seq::Point<int>> expected = {{-1, -1}, {101, -1}, {101, 101}, {-1, 101}};

  // Create TaskData
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  // Create Task
  auto test = std::make_shared<zaitsev_a_jarvis_seq::Jarvis<int>>(taskData);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  out.resize(taskData->outputs_count[0]);
  EXPECT_EQ(expected, out);
}

TEST(zaitsev_a_jarvis_seq_perf, test_task_run) {
  const int size = 100;

  std::vector<zaitsev_a_jarvis_seq::Point<int>> in = zaitsev_a_jarvis_seq::get_random_vector(size);

  in[0] = zaitsev_a_jarvis_seq::Point<int>{-1, -1};
  in[1] = zaitsev_a_jarvis_seq::Point<int>{-1, 101};
  in[2] = zaitsev_a_jarvis_seq::Point<int>{101, 101};
  in[3] = zaitsev_a_jarvis_seq::Point<int>{101, -1};

  std::vector<zaitsev_a_jarvis_seq::Point<int>> out(size, {0, 0});
  std::vector<zaitsev_a_jarvis_seq::Point<int>> expected = {{-1, -1}, {101, -1}, {101, 101}, {-1, 101}};

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<zaitsev_a_jarvis_seq::Jarvis<int>>(taskData);

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

  out.resize(taskData->outputs_count[0]);

  EXPECT_EQ(expected, out);
}