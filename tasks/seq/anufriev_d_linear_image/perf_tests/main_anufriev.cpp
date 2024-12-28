#include <gtest/gtest.h>

#include <chrono>
#include <random>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/anufriev_d_linear_image/include/ops_seq_anufriev.hpp"

std::vector<int> generate_random_image(int rows, int cols) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, 255);
  std::vector<int> image(rows * cols);
  std::generate(image.begin(), image.end(), [&]() { return distrib(gen); });
  return image;
}

#define PERF_TEST_SEQ(test_name, rows_const, cols_const, num_runs, perf_method)    \
  TEST(anufriev_d_linear_image_perf_seq, test_name) {                              \
    int rows = rows_const;                                                         \
    int cols = cols_const;                                                         \
    auto taskData = std::make_shared<ppc::core::TaskData>();                       \
    std::vector<int> input = generate_random_image(rows, cols);                    \
    std::vector<int> output(rows* cols);                                           \
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));          \
    taskData->inputs_count.push_back(input.size());                                \
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));                 \
    taskData->inputs_count.push_back(sizeof(int));                                 \
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));                 \
    taskData->inputs_count.push_back(sizeof(int));                                 \
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));        \
    taskData->outputs_count.push_back(output.size());                              \
    auto task = std::make_shared<anufriev_d_linear_image::SimpleIntSEQ>(taskData); \
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();                       \
    perfAttr->num_running = num_runs;                                              \
    auto start = std::chrono::high_resolution_clock::now();                        \
    perfAttr->current_timer = [&]() {                                              \
      auto end = std::chrono::high_resolution_clock::now();                        \
      std::chrono::duration<double> elapsed = end - start;                         \
      return elapsed.count();                                                      \
    };                                                                             \
    auto perfResults = std::make_shared<ppc::core::PerfResults>();                 \
    auto perf = std::make_shared<ppc::core::Perf>(task);                           \
    perf->perf_method(perfAttr, perfResults);                                      \
    ppc::core::Perf::print_perf_statistic(perfResults);                            \
  }

PERF_TEST_SEQ(LargeImage, 5000, 5000, 1, pipeline_run)

PERF_TEST_SEQ(LargeImageRun, 5000, 5000, 1, task_run)

#undef PERF_TEST_SEQ