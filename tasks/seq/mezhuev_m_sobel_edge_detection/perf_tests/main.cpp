#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <vector>

#include "seq/mezhuev_m_sobel_edge_detection/include/seq.hpp"

TEST(mezhuev_m_sobel_edge_detection, PreProcessingPerformance) {
  size_t width = 1920;
  size_t height = 1080;

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq;

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq::TaskData task_data;

  task_data.width = width;
  task_data.height = height;
  task_data.inputs_count.push_back(width * height);
  task_data.outputs_count.push_back(width * height);
  task_data.inputs.push_back(new uint8_t[width * height]());
  task_data.outputs.push_back(new uint8_t[width * height]());

  auto start = std::chrono::high_resolution_clock::now();
  bool preprocessing_result = sobel_edge_detection_seq.pre_processing(&task_data);
  EXPECT_TRUE(preprocessing_result);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "PreProcessing duration: " << duration.count() << " ms" << std::endl;

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PostProcessingPerformance) {
  size_t width = 1920;
  size_t height = 1080;

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq;

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq::TaskData task_data;

  task_data.width = width;
  task_data.height = height;
  task_data.inputs_count.push_back(width * height);
  task_data.outputs_count.push_back(width * height);
  task_data.inputs.push_back(new uint8_t[width * height]());
  task_data.outputs.push_back(new uint8_t[width * height]());

  EXPECT_TRUE(sobel_edge_detection_seq.pre_processing(&task_data));
  EXPECT_TRUE(sobel_edge_detection_seq.run());

  auto start = std::chrono::high_resolution_clock::now();
  bool postprocessing_result = sobel_edge_detection_seq.post_processing();
  EXPECT_TRUE(postprocessing_result);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "PostProcessing duration: " << duration.count() << " ms" << std::endl;

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}