#include "seq/chistov_a_convex_hull_image_seq/include/image.hpp"

#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"

TEST(chistov_a_convex_hull_image_seq, test_pipeline_run) {
  const int width = 2500;
  const int height = 2500;

  std::vector<int> image(width * height, 1);
  std::vector<int> hull(width * height);

  std::vector<int> expected_hull(width * height, 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        expected_hull[y * width + x] = 1;
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

  auto TestTaskSequential = std::make_shared<chistov_a_convex_hull_image_seq::ConvexHullSEQ>(taskDataSeq);

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

  ASSERT_EQ(hull, expected_hull);
}

TEST(chistov_a_convex_hull_image_seq, test_task_run) {
  const int width = 2500;
  const int height = 2500;

  std::vector<int> image(width * height, 1);
  std::vector<int> hull(width * height);

  std::vector<int> expected_hull(width * height, 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        expected_hull[y * width + x] = 1;
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

  auto TestTaskSequential = std::make_shared<chistov_a_convex_hull_image_seq::ConvexHullSEQ>(taskDataSeq);

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

  ASSERT_EQ(hull, expected_hull);
}
