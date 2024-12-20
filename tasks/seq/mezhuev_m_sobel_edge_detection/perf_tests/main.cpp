#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/mezhuev_m_sobel_edge_detection/include/seq.hpp"

TEST(SobelEdgeDetectionSeqTest, RunPerformanceMultipleExecutions) {
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
  std::cout << "Average execution time for " << num_iterations << " iterations: " << avg_duration << " seconds."
            << std::endl;

  EXPECT_LT(avg_duration, 5.0);
}

TEST(SobelEdgeDetectionSeqTest, RunPerformanceDifferentSizes) {
  const std::vector<size_t> image_sizes = {100, 500, 1000};
  const int num_iterations = 3;
  double total_duration = 0;

  for (size_t size : image_sizes) {
    std::cout << "Running Sobel edge detection for image size " << size << "x" << size << "..." << std::endl;

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs_count.push_back(size);
    task_data->outputs_count.push_back(size);

    auto input_image = new uint8_t[size * size]{0};
    auto output_image = new uint8_t[size * size]{0};

    task_data->inputs.push_back(input_image);
    task_data->outputs.push_back(output_image);

    mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq(task_data);

    for (int i = 0; i < num_iterations; ++i) {
      auto start_time = std::chrono::high_resolution_clock::now();
      EXPECT_TRUE(sobel_edge_detection_seq.run());
      auto end_time = std::chrono::high_resolution_clock::now();
      total_duration += std::chrono::duration<double>(end_time - start_time).count();
    }

    double avg_duration = total_duration / (num_iterations * image_sizes.size());
    std::cout << "Average execution time for image size " << size << "x" << size << ": " << avg_duration << " seconds."
              << std::endl;

    EXPECT_LT(avg_duration, 5.0);

    delete[] input_image;
    delete[] output_image;
  }
}