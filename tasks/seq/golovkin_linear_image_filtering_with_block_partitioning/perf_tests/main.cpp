// Golovkin Maksim Task#3

#include <gtest/gtest.h>

#include <chrono>
#include <random>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/golovkin_linear_image_filtering_with_block_partitioning/include/ops_seq.hpp"

using namespace std;

vector<int> generate_random_images(int rows, int cols) {
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<> distrib(0, 255);
  vector<int> image(rows * cols);
  generate(image.begin(), image.end(), [&]() { return distrib(gen); });
  return image;
}

#define PERF_TEST_SEQ_BLOCK(test_name, rows_const, cols_const, block_size, num_runs)                          \
  TEST(golovkin_linear_image_filtering_with_block_partitioning, test_name) {                                  \
    int rows = rows_const;                                                                                    \
    int cols = cols_const;                                                                                    \
    auto taskData = make_shared<ppc::core::TaskData>();                                                       \
    vector<int> input = generate_random_images(rows, cols);                                                   \
    vector<int> output(rows* cols);                                                                           \
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));                                     \
    taskData->inputs_count.push_back(input.size());                                                           \
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));                                            \
    taskData->inputs_count.push_back(sizeof(int));                                                            \
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));                                            \
    taskData->inputs_count.push_back(sizeof(int));                                                            \
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));                                   \
    taskData->outputs_count.push_back(output.size());                                                         \
    auto task = make_shared<golovkin_linear_image_filtering_with_block_partitioning::SimpleIntSEQ>(taskData); \
    auto perfAttr = make_shared<ppc::core::PerfAttr>();                                                       \
    perfAttr->num_running = num_runs;                                                                         \
    auto start = chrono::high_resolution_clock::now();                                                        \
    perfAttr->current_timer = [&]() {                                                                         \
      auto end = chrono::high_resolution_clock::now();                                                        \
      chrono::duration<double> elapsed = end - start;                                                         \
      return elapsed.count();                                                                                 \
    };                                                                                                        \
    auto perfResults = make_shared<ppc::core::PerfResults>();                                                 \
    auto perf = make_shared<ppc::core::Perf>(task);                                                           \
    perf->pipeline_run(perfAttr, perfResults);                                                                \
    ppc::core::Perf::print_perf_statistic(perfResults);                                                       \
  }

PERF_TEST_SEQ_BLOCK(Block16x16, 500, 500, 16, 5)
PERF_TEST_SEQ_BLOCK(Block32x32, 500, 500, 32, 5)
PERF_TEST_SEQ_BLOCK(Block64x64, 500, 500, 64, 5)

#undef PERF_TEST_SEQ
#undef PERF_TEST_SEQ_BLOCK