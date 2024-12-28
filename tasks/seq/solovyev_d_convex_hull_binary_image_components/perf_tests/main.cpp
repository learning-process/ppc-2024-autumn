#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/solovyev_d_convex_hull_binary_image_components/include/header.hpp"
namespace solovyev_d_convex_hull_binary_image_components_seq {
std::vector<int> generateVector(int sz) {
  std::vector<int> vec(sz);
  for (int i = 0; i < sz / 4; i++) {
    vec[i] = 0;
  }
  for (int i = sz / 4; i < 2 * sz / 4; i++) {
    vec[i] = 1;
  }
  for (int i = 2 * sz / 4; i < 3 * sz / 4; i++) {
    vec[i] = 0;
  }
  for (int i = 3 * sz / 4; i < sz; i++) {
    vec[i] = 1;
  }
  return vec;
}
}  // namespace solovyev_d_convex_hull_binary_image_components_seq
TEST(solovyev_d_convex_hull_binary_image_components_seq, test_pipeline_run) {
  int dimX = 500;
  int dimY = 500;
  // Create data
  std::vector<int> in = solovyev_d_convex_hull_binary_image_components_seq::generateVector(500 * 500);
  std::vector<std::vector<int>> expected = {{499, 249, 499, 125, 0, 125, 0, 249}, {499, 499, 499, 375, 0, 375, 0, 499}};
  std::vector<std::vector<int>> out = {std::vector<int>(expected[0].size(), 0),
                                       std::vector<int>(expected[1].size(), 0)};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimX));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimY));
  taskDataSeq->inputs_count.emplace_back(in.size());
  for (size_t i = 0; i < out.size(); i++) {
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
    taskDataSeq->outputs_count.emplace_back(out.size());
  }

  // Create Task
  auto testTaskSequential =
      std::make_shared<solovyev_d_convex_hull_binary_image_components_seq::ConvexHullBinaryImageComponentsSequential>(
          taskDataSeq);
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
  ASSERT_EQ(expected, out);
}

TEST(solovyev_d_convex_hull_binary_image_components_seq, test_task_run) {
  int dimX = 500;
  int dimY = 500;
  // Create data
  std::vector<int> in = solovyev_d_convex_hull_binary_image_components_seq::generateVector(500 * 500);
  std::vector<std::vector<int>> expected = {{499, 249, 499, 125, 0, 125, 0, 249}, {499, 499, 499, 375, 0, 375, 0, 499}};
  std::vector<std::vector<int>> out = {std::vector<int>(expected[0].size(), 0),
                                       std::vector<int>(expected[1].size(), 0)};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimX));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimY));
  taskDataSeq->inputs_count.emplace_back(in.size());
  for (size_t i = 0; i < out.size(); i++) {
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
    taskDataSeq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  auto testTaskSequential =
      std::make_shared<solovyev_d_convex_hull_binary_image_components_seq::ConvexHullBinaryImageComponentsSequential>(
          taskDataSeq);

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
  ASSERT_EQ(expected, out);
}
