#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#include "seq/mezhuev_m_sobel_edge_detection/include/seq.hpp"

TEST(mezhuev_m_sobel_edge_detection, ValidationTest_ValidData) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(25);
  task_data->outputs_count.push_back(25);
  task_data->inputs.push_back(new uint8_t[25]{0});
  task_data->outputs.push_back(new uint8_t[25]{0});

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq(task_data);

  EXPECT_TRUE(sobel_edge_detection_seq.validation());

  delete[] task_data->inputs[0];
  delete[] task_data->outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, ValidationTest_NullInputOrOutputBuffer) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(nullptr);
  task_data->outputs.push_back(new uint8_t[25]{0});

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq(task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.validation());

  delete[] task_data->outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, ValidationTest_EmptyInputOrOutputBuffer) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(new uint8_t[25]{0});
  task_data->outputs.push_back(nullptr);

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq(task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.validation());

  delete[] task_data->inputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, ValidationTest_MultipleInputsOrOutputs) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(25);
  task_data->outputs_count.push_back(25);
  task_data->inputs.push_back(new uint8_t[25]{0});
  task_data->outputs.push_back(new uint8_t[25]{0});
  task_data->inputs.push_back(new uint8_t[25]{0});

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq(task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.validation());

  delete[] task_data->inputs[0];
  delete[] task_data->inputs[1];
  delete[] task_data->outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, ValidationTest_MismatchInInputAndOutputBufferSizes) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(25);
  task_data->outputs_count.push_back(24);
  task_data->inputs.push_back(new uint8_t[25]{0});
  task_data->outputs.push_back(new uint8_t[25]{0});

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq(task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.validation());

  delete[] task_data->inputs[0];
  delete[] task_data->outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, ValidationTest_EmptyTaskData) {
  auto task_data = std::make_shared<ppc::core::TaskData>();

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq(task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.validation());
}

TEST(mezhuev_m_sobel_edge_detection, PreProcessingTest_ValidData) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(5);
  task_data->inputs.push_back(new uint8_t[5 * 5]{0});
  task_data->outputs.push_back(new uint8_t[5 * 5]{0});

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq(task_data);

  EXPECT_TRUE(sobel_edge_detection_seq.pre_processing());

  delete[] task_data->inputs[0];
  delete[] task_data->outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, RunTest_NullBuffers) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(5);
  task_data->inputs.push_back(nullptr);
  task_data->outputs.push_back(nullptr);

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq(task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.run());
}

TEST(mezhuev_m_sobel_edge_detection, RunTest_SimpleEdgeDetection) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(5);
  task_data->inputs.push_back(
      new uint8_t[5 * 5]{0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  task_data->outputs.push_back(new uint8_t[5 * 5]{0});

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq(task_data);

  EXPECT_TRUE(sobel_edge_detection_seq.run());

  EXPECT_GT(task_data->outputs[0][6], 0);
  EXPECT_GT(task_data->outputs[0][7], 0);
  EXPECT_GT(task_data->outputs[0][8], 0);

  delete[] task_data->inputs[0];
  delete[] task_data->outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, RunTest_LargerImage) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(10);
  task_data->inputs.push_back(new uint8_t[10 * 10]{
      0, 0, 0,   0,   0,   0, 0, 0,   0,   0,   0, 0, 0,   0,   0,   0, 0, 0,   0,   0,   0, 0, 255, 255, 255,
      0, 0, 0,   0,   0,   0, 0, 255, 255, 255, 0, 0, 0,   0,   0,   0, 0, 255, 255, 255, 0, 0, 0,   0,   0,
      0, 0, 255, 255, 255, 0, 0, 0,   0,   0,   0, 0, 255, 255, 255, 0, 0, 0,   0,   0,   0, 0, 255, 255, 255,
      0, 0, 0,   0,   0,   0, 0, 255, 255, 255, 0, 0, 0,   0,   0,   0, 0, 0,   0,   0,   0, 0, 0,   0,   0});
  task_data->outputs.push_back(new uint8_t[10 * 10]{0});

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq(task_data);

  EXPECT_TRUE(sobel_edge_detection_seq.run());

  EXPECT_GT(task_data->outputs[0][11], 0);

  delete[] task_data->inputs[0];
  delete[] task_data->outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PostProcessingTest_NullOutputBuffer) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->outputs.push_back(nullptr);
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq(task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.post_processing());
}

TEST(mezhuev_m_sobel_edge_detection, PostProcessingTest_ValidOutputBuffer_WithZeroValues) {
  auto task_data = std::make_shared<ppc::core::TaskData>();

  size_t output_size = 5;
  task_data->outputs_count.push_back(output_size);
  uint8_t* output_data = new uint8_t[output_size]{0, 255, 255, 0, 0};

  task_data->outputs.push_back(output_data);

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq(task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.post_processing());

  delete[] output_data;
}

TEST(mezhuev_m_sobel_edge_detection, PostProcessingTest_ValidOutputBuffer_WithNonZeroValues) {
  auto task_data = std::make_shared<ppc::core::TaskData>();

  size_t output_size = 5;
  task_data->outputs_count.push_back(output_size);
  uint8_t* output_data = new uint8_t[output_size]{255, 255, 255, 255, 255};

  task_data->outputs.push_back(output_data);

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq(task_data);

  EXPECT_TRUE(sobel_edge_detection_seq.post_processing());

  delete[] output_data;
}