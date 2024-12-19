#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#include "seq/mezhuev_m_sobel_edge_detection/include/seq.hpp"

TEST(mezhuev_m_sobel_edge_detection, ValidationTestValidData) {
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq;

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;

  task_data.inputs_count.push_back(25);
  task_data.outputs_count.push_back(25);

  task_data.inputs.push_back(new uint8_t[25]{0});
  task_data.outputs.push_back(new uint8_t[25]{0});

  sobel_edge_detection_seq.setTaskData(&task_data);

  EXPECT_TRUE(sobel_edge_detection_seq.validation());

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, validationtest_nullInputOrOutputBuffer) {
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq;

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;
  task_data.inputs.push_back(nullptr);
  task_data.outputs.push_back(new uint8_t[25]{0});

  sobel_edge_detection_seq.setTaskData(&task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.validation());

  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, ValidationTest_MultipleInputsOrOutputs) {
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq;

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;
  task_data.inputs_count.push_back(25);
  task_data.outputs_count.push_back(25);

  task_data.inputs.push_back(new uint8_t[25]{0});
  task_data.outputs.push_back(new uint8_t[25]{0});
  task_data.inputs.push_back(new uint8_t[25]{0});

  sobel_edge_detection_seq.setTaskData(&task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.validation());

  delete[] task_data.inputs[0];
  delete[] task_data.inputs[1];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PreProcessingInvalidData) {
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq;

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;

  sobel_edge_detection_seq.setTaskData(&task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.pre_processing(&task_data));
}

TEST(mezhuev_m_sobel_edge_detection, PreProcessingMismatchedInputsAndOutputs) {
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq;

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;
  task_data.inputs_count.push_back(25);
  task_data.outputs_count.push_back(24);

  task_data.inputs.push_back(new uint8_t[25]{0});
  task_data.outputs.push_back(new uint8_t[25]{0});

  sobel_edge_detection_seq.setTaskData(&task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.pre_processing(&task_data));

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PreProcessingNullInputOrOutputBuffer) {
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq;

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;
  task_data.inputs.push_back(nullptr);
  task_data.outputs.push_back(new uint8_t[25]{0});

  sobel_edge_detection_seq.setTaskData(&task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.pre_processing(&task_data));

  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PreProcessingMultipleInputsOrOutputs) {
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq;

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;
  task_data.inputs_count.push_back(25);
  task_data.outputs_count.push_back(25);

  task_data.inputs.push_back(new uint8_t[25]{0});
  task_data.outputs.push_back(new uint8_t[25]{0});
  task_data.inputs.push_back(new uint8_t[25]{0});

  sobel_edge_detection_seq.setTaskData(&task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.pre_processing(&task_data));

  delete[] task_data.inputs[0];
  delete[] task_data.inputs[1];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PostProcessingInvalidValue) {
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq;

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;
  task_data.inputs_count.push_back(25);
  task_data.outputs_count.push_back(25);

  task_data.inputs.push_back(
      new uint8_t[25]{0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0});
  task_data.outputs.push_back(new uint8_t[25]{0});

  sobel_edge_detection_seq.setTaskData(&task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.post_processing());

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PostProcessingNullOutputBuffer) {
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq;

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;
  task_data.inputs_count.push_back(25);
  task_data.outputs_count.push_back(25);

  task_data.inputs.push_back(
      new uint8_t[25]{0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0});
  task_data.outputs.push_back(nullptr);

  sobel_edge_detection_seq.setTaskData(&task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.post_processing());

  delete[] task_data.inputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, RunNullptrTest) {
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel;

  sobel.setTaskData(nullptr);

  EXPECT_FALSE(sobel.run());
}

TEST(mezhuev_m_sobel_edge_detection, EdgePixelsTest) {
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel;

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq::TaskData task_data_instance;
  task_data_instance.width = 5;
  task_data_instance.height = 5;
  task_data_instance.inputs.resize(1);
  task_data_instance.outputs.resize(1);

  std::vector<uint8_t> input_image = {0,   0, 0, 0,   0,   0,   255, 255, 255, 0, 0, 255, 255,
                                      255, 0, 0, 255, 255, 255, 0,   0,   0,   0, 0, 0};
  task_data_instance.inputs[0] = input_image.data();

  std::vector<uint8_t> output_image(25, 0);
  task_data_instance.outputs[0] = output_image.data();

  sobel.setTaskData(&task_data_instance);

  EXPECT_TRUE(sobel.run());

  EXPECT_EQ(output_image[0], 0);
  EXPECT_EQ(output_image[4], 0);
  EXPECT_EQ(output_image[20], 0);
  EXPECT_EQ(output_image[24], 0);
}

TEST(mezhuev_m_sobel_edge_detection, OutputSizeTest) {
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel;

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq::TaskData task_data_instance;
  task_data_instance.width = 5;
  task_data_instance.height = 5;
  task_data_instance.inputs.resize(1);
  task_data_instance.outputs.resize(1);

  std::vector<uint8_t> input_image = {0,   0, 0, 0,   0,   0,   255, 255, 255, 0, 0, 255, 255,
                                      255, 0, 0, 255, 255, 255, 0,   0,   0,   0, 0, 0};
  task_data_instance.inputs[0] = input_image.data();

  std::vector<uint8_t> output_image(25, 0);
  task_data_instance.outputs[0] = output_image.data();

  sobel.setTaskData(&task_data_instance);

  EXPECT_TRUE(sobel.run());

  EXPECT_EQ(output_image.size(), task_data_instance.width * task_data_instance.height);
}

TEST(mezhuev_m_sobel_edge_detection, Handle1x1Image) {
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = 1;
  task_data.height = 1;
  uint8_t pixel = 255;
  task_data.inputs.push_back(&pixel);
  task_data.inputs_count.push_back(1);
  uint8_t output = 0;
  task_data.outputs.push_back(&output);
  task_data.outputs_count.push_back(1);

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel;
  sobel.setTaskData(&task_data);

  bool result = sobel.run();
  EXPECT_TRUE(result);
  EXPECT_EQ(output, 0);
}

TEST(mezhuev_m_sobel_edge_detection, HandleLargeImage) {
  const int size = 1024;
  std::vector<uint8_t> input(size * size, 255);
  std::vector<uint8_t> output(size * size, 0);

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = size;
  task_data.height = size;
  task_data.inputs.push_back(input.data());
  task_data.inputs_count.push_back(input.size());
  task_data.outputs.push_back(output.data());
  task_data.outputs_count.push_back(output.size());

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel;
  sobel.setTaskData(&task_data);

  bool result = sobel.run();
  EXPECT_TRUE(result);
}

TEST(mezhuev_m_sobel_edge_detection, HandleImageWithNoise) {
  const int size = 5;
  uint8_t input[25] = {255, 255, 255, 255, 255, 255, 0,   0, 0, 255, 255, 0,  0,
                       0,   255, 255, 255, 255, 255, 255, 0, 0, 255, 255, 255};
  uint8_t output[25] = {};

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = size;
  task_data.height = size;
  task_data.inputs.push_back(input);
  task_data.inputs_count.push_back(25);
  task_data.outputs.push_back(output);
  task_data.outputs_count.push_back(25);

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel;
  sobel.setTaskData(&task_data);

  bool result = sobel.run();
  EXPECT_TRUE(result);

  EXPECT_GT(output[6], 0);
}