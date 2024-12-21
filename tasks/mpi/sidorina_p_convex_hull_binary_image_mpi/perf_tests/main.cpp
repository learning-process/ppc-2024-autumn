#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/sidorina_p_convex_hull_binary_image_mpi/include/ops_mpi.hpp"

TEST(sidorina_p_convex_hull_binary_image_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  const int width = 1500;
  const int height = 1500;

  std::vector<int> image(width * height, 1);
  std::vector<int> hull(width * height, 0);
  std::vector<int> ref(width * height, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (int j = 0; j < height; j++) {
      ref[j * width + 0] = 1;
      ref[j * width + width - 1] = 1;
      if (j == 0 || j == height - 1) {
        for (int i = 1; i < width - 1; i++) {
          ref[j * width + i] = 1;
        }
      }
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);
  }

  auto TestTaskPar = std::make_shared<sidorina_p_convex_hull_binary_image_mpi::ConvexHullBinImgMpi>(taskDataPar);

  ASSERT_TRUE(TestTaskPar->validation());
  TestTaskPar->pre_processing();
  TestTaskPar->run();
  TestTaskPar->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TestTaskPar);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    ASSERT_EQ(hull, ref);
  }
}

TEST(sidorina_p_convex_hull_binary_image_mpi, test_task_run) {
  boost::mpi::communicator world;
  const int width = 1500;
  const int height = 1500;

  std::vector<int> image(width * height, 1);
  std::vector<int> hull(width * height, 0);
  std::vector<int> ref(width * height, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (int j = 0; j < height; j++) {
      ref[j * width + 0] = 1;
      ref[j * width + width - 1] = 1;
      if (j == 0 || j == height - 1) {
        for (int i = 1; i < width - 1; i++) {
          ref[j * width + i] = 1;
        }
      }
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);
  }

  auto TestTaskPar = std::make_shared<sidorina_p_convex_hull_binary_image_mpi::ConvexHullBinImgMpi>(taskDataPar);

  ASSERT_TRUE(TestTaskPar->validation());
  TestTaskPar->pre_processing();
  TestTaskPar->run();
  TestTaskPar->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TestTaskPar);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    ASSERT_EQ(hull, ref);
  }
}
