#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/sidorina_p_convex_hull_binary_image_seq/include/ops_seq.hpp"

TEST(sidorina_p_convex_hull_binary_image_seq, test_pipeline_run) {
  const int width = 1500;
  const int height = 1500;

  std::vector<int> image(width * height, 1);
  std::vector<int> hull(width * height);

  std::vector<int> ref(width * height, 0);
  for (int j = 0; j < height; j++) {
    ref[j * width + 0] = 1;
    ref[j * width + width - 1] = 1;
    if (j == 0 || j == height - 1) {
      for (int i = 1; i < width - 1; i++) {
        ref[j * width + i] = 1;
      }
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  auto TestTaskSequential = std::make_shared<sidorina_p_convex_hull_binary_image_seq::ConvexHullBinImgSeq>(taskDataSeq);

  ASSERT_TRUE(TestTaskSequential->validation());
  TestTaskSequential->pre_processing();
  TestTaskSequential->run();
  TestTaskSequential->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TestTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(ref, hull);
}

TEST(sidorina_p_convex_hull_binary_image_seq, test_task_run) {
  const int width = 1500;
  const int height = 1500;

  std::vector<int> image(width * height, 1);
  std::vector<int> hull(width * height);

  std::vector<int> ref(width * height, 0);
  for (int j = 0; j < height; j++) {
    ref[j * width + 0] = 1;
    ref[j * width + width - 1] = 1;
    if (j == 0 || j == height - 1) {
      for (int i = 1; i < width - 1; i++) {
        ref[j * width + i] = 1;
      }
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  auto TestTaskSequential = std::make_shared<sidorina_p_convex_hull_binary_image_seq::ConvexHullBinImgSeq>(taskDataSeq);

  ASSERT_TRUE(TestTaskSequential->validation());
  TestTaskSequential->pre_processing();
  TestTaskSequential->run();
  TestTaskSequential->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TestTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(ref, hull);
}