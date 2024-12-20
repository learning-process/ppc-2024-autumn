#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/mezhuev_m_sobel_edge_detection/include/seq.hpp"

TEST(mezhuev_m_sobel_edge_detection, RunPerformanceMultipleExecutions) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  size_t image_size = 1000;
  task_data->inputs_count.push_back(image_size);
  task_data->outputs_count.push_back(image_size);
  auto input_image = std::make_unique<uint8_t[]>(image_size * image_size);
  auto output_image = std::make_unique<uint8_t[]>(image_size * image_size);

  task_data->inputs.push_back(input_image.get());
  task_data->outputs.push_back(output_image.get());

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq(task_data);

  const int num_iterations = 5;
  double total_duration = 0;

  for (int i = 0; i < num_iterations; ++i) {
    auto start_time = std::chrono::high_resolution_clock::now();
    EXPECT_TRUE(sobel_edge_detection_seq.run());
    auto end_time = std::chrono::high_resolution_clock::now();
    total_duration += std::chrono::duration<double>(end_time - start_time).count();
  }

  double avg_duration = total_duration / num_iterations;

  EXPECT_LT(avg_duration, 5.0);
}

TEST(mezhuev_m_sobel_edge_detection, RunPerformanceDifferentSizes) {
  const std::vector<size_t> image_sizes = {100, 500, 1000};
  const int num_iterations = 3;
  double total_duration = 0;

  for (size_t size : image_sizes) {

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs_count.push_back(size);
    task_data->outputs_count.push_back(size);

    auto input_image = std::make_unique<uint8_t[]>(size * size);
    auto output_image = std::make_unique<uint8_t[]>(size * size);

    task_data->inputs.push_back(input_image.get());
    task_data->outputs.push_back(output_image.get());

    mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq(task_data);

    for (int i = 0; i < num_iterations; ++i) {
      auto start_time = std::chrono::high_resolution_clock::now();
      EXPECT_TRUE(sobel_edge_detection_seq.run());
      auto end_time = std::chrono::high_resolution_clock::now();
      total_duration += std::chrono::duration<double>(end_time - start_time).count();
    }

    double avg_duration = total_duration / (num_iterations * image_sizes.size());

    EXPECT_LT(avg_duration, 5.0);
  }
}