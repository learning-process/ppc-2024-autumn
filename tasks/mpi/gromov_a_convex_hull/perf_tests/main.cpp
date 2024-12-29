#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/gromov_a_convex_hull/include/ops_mpi.hpp"

TEST(gromov_a_convex_hull_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  const int width = 400;
  const int height = 400;

  std::vector<int> image(width * height, 1);
  std::vector<int> hull(width * height);
  std::vector<int> expected_hull(width * height, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
          expected_hull[y * width + x] = 1;
        }
      }
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);
  }

  auto testMpiTaskParallel = std::make_shared<gromov_a_convex_hull_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel->validation());
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(hull, expected_hull);
  }
}

TEST(gromov_a_convex_hull_mpi, test_task_run) {
  boost::mpi::communicator world;
  const int width = 400;
  const int height = 400;

  std::vector<int> image(width * height, 1);
  std::vector<int> hull(width * height);
  std::vector<int> expected_hull(width * height, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
          expected_hull[y * width + x] = 1;
        }
      }
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);
  }

  auto testMpiTaskParallel = std::make_shared<gromov_a_convex_hull_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel->validation());
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(hull, expected_hull);
  }
}
