// Golovkin Maksim Task#3

#include <gtest/gtest.h>

#include <chrono>
#include <numeric>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/golovkin_linear_image_filtering_with_block_partitioning/include/ops_seq.hpp"

// Генерация случайного изображения
std::vector<int> generate_random_image(int rows, int cols) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, 255);
  std::vector<int> image(rows * cols);
  std::generate(image.begin(), image.end(), [&]() { return distrib(gen); });
  return image;
}

// Тесты с использованием разных размеров блоков для фильтрации с блочным разбиением
#define PERF_TEST_SEQ_BLOCK(test_name, rows_const, cols_const, block_size, num_runs)                               \
  TEST(golovkin_linear_image_filtering_with_block_partitioning, test_name) {                                       \
    int rows = rows_const;                                                                                         \
    int cols = cols_const;                                                                                         \
    auto taskData = std::make_shared<ppc::core::TaskData>();                                                       \
    std::vector<int> input = generate_random_image(rows, cols);                                                    \
    std::vector<int> output(rows* cols);                                                                           \
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));                                          \
    taskData->inputs_count.push_back(input.size());                                                                \
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));                                                 \
    taskData->inputs_count.push_back(sizeof(int));                                                                 \
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));                                                 \
    taskData->inputs_count.push_back(sizeof(int));                                                                 \
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));                                        \
    taskData->outputs_count.push_back(output.size());                                                              \
    auto task = std::make_shared<golovkin_linear_image_filtering_with_block_partitioning::SimpleIntSEQ>(taskData); \
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();                                                       \
    perfAttr->num_running = num_runs;                                                                              \
    auto start = std::chrono::high_resolution_clock::now();                                                        \
    perfAttr->current_timer = [&]() {                                                                              \
      auto end = std::chrono::high_resolution_clock::now();                                                        \
      std::chrono::duration<double> elapsed = end - start;                                                         \
      return elapsed.count();                                                                                      \
    };                                                                                                             \
    auto perfResults = std::make_shared<ppc::core::PerfResults>();                                                 \
    auto perf = std::make_shared<ppc::core::Perf>(task);                                                           \
    perf->pipeline_run(perfAttr, perfResults);                                                                     \
    ppc::core::Perf::print_perf_statistic(perfResults);                                                            \
  }

// Тесты с блоками 16x16, 32x32 и 64x64
PERF_TEST_SEQ_BLOCK(Block16x16, 500, 500, 16, 5)  // 500x500 изображение, блоки 16x16, 5 запусков
PERF_TEST_SEQ_BLOCK(Block32x32, 500, 500, 32, 5)  // 500x500 изображение, блоки 32x32, 5 запусков
PERF_TEST_SEQ_BLOCK(Block64x64, 500, 500, 64, 5)  // 500x500 изображение, блоки 64x64, 5 запусков

#undef PERF_TEST_SEQ
#undef PERF_TEST_SEQ_BLOCK