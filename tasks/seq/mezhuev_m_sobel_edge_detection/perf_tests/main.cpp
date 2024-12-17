#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <vector>
#include <cstdlib>

#include "seq/mezhuev_m_sobel_edge_detection/include/seq.hpp"

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

TEST(mezhuev_m_sobel_edge_detection, PreProcessingValidInputs) {
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq;

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;
  task_data.inputs_count.push_back(25);
  task_data.outputs_count.push_back(25);

  task_data.inputs.push_back(new uint8_t[25]{0});
  task_data.outputs.push_back(new uint8_t[25]{0});

  sobel_edge_detection_seq.setTaskData(&task_data);

  EXPECT_TRUE(sobel_edge_detection_seq.pre_processing(&task_data));

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, RunPerformance) {
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq;

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;
  task_data.inputs_count.push_back(25);
  task_data.outputs_count.push_back(25);

  task_data.inputs.push_back(new uint8_t[25]{0});
  task_data.outputs.push_back(new uint8_t[25]{0});

  sobel_edge_detection_seq.setTaskData(&task_data);

  EXPECT_TRUE(sobel_edge_detection_seq.pre_processing(&task_data));

  EXPECT_TRUE(sobel_edge_detection_seq.run());

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}