// Golovkin Maksim Task#3
#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/timer.hpp>
#include <numeric>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/golovkin_linear_image_filtering_with_block_partitioning/include/ops_mpi.hpp"

static std::vector<int> generate_random_image(int width, int height, int seed = 123) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<int> img(width * height);
  for (auto &val : img) {
    val = dist(gen);
  }
  return img;
}

#define PERF_TEST_IMAGE(test_name, W, H, num_runs, perf_method)                                                      \
  TEST(golovkin_linear_image_filtering_with_block_partitioning_perf_mpi, test_name) {                                \
    boost::mpi::communicator world;                                                                                  \
    int width = (W);                                                                                                 \
    int height = (H);                                                                                                \
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();                         \
    std::vector<int> input_data;                                                                                     \
    std::vector<int> output_data;                                                                                    \
    if (world.size() < 5 || world.rank() >= 4) {                                                                     \
      input_data = generate_random_image(width, height);                                                             \
      output_data.resize(width *height, 0);                                                                          \
      taskData->inputs.push_back(reinterpret_cast<uint8_t *>(input_data.data()));                                    \
      taskData->inputs_count.push_back(input_data.size() * sizeof(int));                                             \
                                                                                                                     \
      taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&width));                                               \
      taskData->inputs_count.push_back(sizeof(int));                                                                 \
                                                                                                                     \
      taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&height));                                              \
      taskData->inputs_count.push_back(sizeof(int));                                                                 \
                                                                                                                     \
      taskData->outputs.push_back(reinterpret_cast<uint8_t *>(output_data.data()));                                  \
      taskData->outputs_count.push_back(output_data.size() * sizeof(int));                                           \
    }                                                                                                                \
    auto task = std::make_shared<golovkin_linear_image_filtering_with_block_partitioning::SimpleBlockMPI>(taskData); \
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();                                                         \
    perfAttr->num_running = num_runs;                                                                                \
    boost::mpi::timer current_timer;                                                                                 \
    perfAttr->current_timer = [&]() { return current_timer.elapsed(); };                                             \
    auto perfResults = std::make_shared<ppc::core::PerfResults>();                                                   \
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);                                                     \
    perfAnalyzer->perf_method(perfAttr, perfResults);                                                                \
    if (world.size() < 5 || world.rank() >= 4) {                                                                     \
      ppc::core::Perf::print_perf_statistic(perfResults);                                                            \
      ASSERT_LE(perfResults->time_sec, ppc::core::PerfResults::MAX_TIME);                                            \
    }                                                                                                                \
  }

PERF_TEST_IMAGE(SmallImagePerf, 100, 80, 5, pipeline_run)
PERF_TEST_IMAGE(MediumImagePerf, 1000, 800, 3, pipeline_run)
PERF_TEST_IMAGE(LargeImagePerf, 2000, 2000, 1, pipeline_run)

PERF_TEST_IMAGE(SmallImageTaskRunPerf, 100, 80, 5, task_run)
PERF_TEST_IMAGE(MediumImageTaskRunPerf, 1000, 800, 3, task_run)
PERF_TEST_IMAGE(LargeImageTaskRunPerf, 2000, 2000, 1, task_run)

#undef PERF_TEST_IMAGE