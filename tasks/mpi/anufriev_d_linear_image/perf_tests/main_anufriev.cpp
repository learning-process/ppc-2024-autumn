#include <gtest/gtest.h>
#include <boost/mpi.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>
#include <numeric>
#include <algorithm>

#include "mpi/anufriev_d_linear_image/include/ops_mpi_anufriev.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task_data.hpp"

// Генерация случайного изображения
std::vector<int> generate_random_image(int width, int height) {
    std::vector<int> img(width*height);
    std::mt19937 gen(123);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto &val : img) {
        val = dist(gen);
    }
    return img;
}

#define PERF_TEST_IMAGE(test_name, width, height, num_runs, perf_method)                       \
TEST(anufriev_d_linear_image_perf, test_name) {                                               \
    boost::mpi::communicator world;                                                           \
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();  \
    std::vector<int> input_data;                                                              \
    std::vector<int> output_data;                                                             \
    if (world.rank() == 0) {                                                                  \
      input_data = generate_random_image(width, height);                                      \
      taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));              \
      taskData->inputs_count.push_back(input_data.size());                                    \
      /* Доп. параметры: width, height */                                                     \
      taskData->inputs_count.push_back(width);                                                \
      taskData->inputs_count.push_back(height);                                               \
      output_data.resize(input_data.size());                                                  \
      taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));            \
      taskData->outputs_count.push_back(output_data.size());                                  \
    }                                                                                         \
    auto task = std::make_shared<anufriev_d_linear_image::SimpleIntMPI>(taskData);            \
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();                                  \
    perfAttr->num_running = num_runs;                                                         \
    const boost::mpi::timer current_timer;                                                    \
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };                        \
    auto perfResults = std::make_shared<ppc::core::PerfResults>();                            \
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);                              \
    perfAnalyzer->perf_method(perfAttr, perfResults);                                         \
    if (world.rank() == 0) {                                                                  \
      ppc::core::Perf::print_perf_statistic(perfResults);                                     \
    }                                                                                         \
}

PERF_TEST_IMAGE(SmallImagePerf, 100, 80, 5, pipeline_run)
PERF_TEST_IMAGE(MediumImagePerf, 1000, 800, 3, pipeline_run)
PERF_TEST_IMAGE(LargeImagePerf, 2000, 2000, 1, pipeline_run)

PERF_TEST_IMAGE(SmallImageTaskRunPerf, 100, 80, 5, task_run)
PERF_TEST_IMAGE(MediumImageTaskRunPerf, 1000, 800, 3, task_run)
PERF_TEST_IMAGE(LargeImageTaskRunPerf, 2000, 2000, 1, task_run)

#undef PERF_TEST_IMAGE