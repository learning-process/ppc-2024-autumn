#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "mpi/mezhuev_m_sobel_edge_detection/include/ops_mpi.hpp"

TEST(mezhuev_m_sobel_edge_detection, PreProcessingInvalidData) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(nullptr);
  task_data.outputs.push_back(new uint8_t[10000]{0});
  task_data.inputs_count.push_back(1);
  task_data.outputs_count.push_back(1);

  EXPECT_FALSE(sobel_edge_detection.pre_processing(&task_data));

  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PostProcessingInvalidData) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.outputs.push_back(nullptr);
  task_data.outputs_count.push_back(1);

  EXPECT_FALSE(sobel_edge_detection.post_processing());
}

TEST(mezhuev_m_sobel_edge_detection, ValidationWithEmptyData) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(nullptr);
  task_data.outputs.push_back(nullptr);
  task_data.inputs_count.push_back(0);
  task_data.outputs_count.push_back(0);

  EXPECT_FALSE(sobel_edge_detection.validation());
}

TEST(mezhuev_m_sobel_edge_detection, PostProcessingEmptyOutput) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.outputs.push_back(new uint8_t[0]);
  task_data.outputs_count.push_back(1);

  EXPECT_FALSE(sobel_edge_detection.post_processing());

  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PostProcessingZeroOutput) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.outputs.push_back(new uint8_t[10000]{0});
  task_data.outputs_count.push_back(10000);

  EXPECT_FALSE(sobel_edge_detection.post_processing());

  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PreProcessingInvalidInputCount) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(new uint8_t[10000]{0});
  task_data.outputs.push_back(new uint8_t[10000]{0});
  task_data.inputs_count.push_back(1);
  task_data.outputs_count.push_back(10000);

  EXPECT_FALSE(sobel_edge_detection.pre_processing(&task_data));

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PreProcessingInvalidDimensions) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(new uint8_t[10000]{0});
  task_data.outputs.push_back(new uint8_t[10000]{0});
  task_data.inputs_count.push_back(10000);
  task_data.outputs_count.push_back(10000);
  task_data.width = 0;
  task_data.height = 100;

  EXPECT_FALSE(sobel_edge_detection.pre_processing(&task_data));

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PreProcessingEmptyImage) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(new uint8_t[0]);
  task_data.outputs.push_back(new uint8_t[0]);
  task_data.inputs_count.push_back(0);
  task_data.outputs_count.push_back(0);
  task_data.width = 0;
  task_data.height = 0;

  EXPECT_FALSE(sobel_edge_detection.pre_processing(&task_data));

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PreProcessingEmptyInputAndOutput) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(nullptr);
  task_data.outputs.push_back(nullptr);
  task_data.inputs_count.push_back(0);
  task_data.outputs_count.push_back(0);
  task_data.width = 0;
  task_data.height = 0;

  EXPECT_FALSE(sobel_edge_detection.pre_processing(&task_data));
}

TEST(mezhuev_m_sobel_edge_detection, InvalidDataBetweenProcesses) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(new uint8_t[10000]{0});
  task_data.outputs.push_back(new uint8_t[10000]{0});
  task_data.inputs_count.push_back(10000);
  task_data.outputs_count.push_back(5000);  // Ошибка в количестве данных для выходных данных
  task_data.width = 100;
  task_data.height = 100;

  EXPECT_FALSE(sobel_edge_detection.pre_processing(&task_data));

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, InvalidImageDimensions) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(new uint8_t[10000]{0});
  task_data.outputs.push_back(new uint8_t[10000]{0});
  task_data.inputs_count.push_back(10000);
  task_data.outputs_count.push_back(10000);
  task_data.width = 1;  // Недопустимая ширина изображения
  task_data.height = 100;

  EXPECT_FALSE(sobel_edge_detection.pre_processing(&task_data));

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, NullPointerInputData) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(nullptr);  // Пустой указатель на входные данные
  task_data.outputs.push_back(new uint8_t[10000]{0});
  task_data.inputs_count.push_back(10000);
  task_data.outputs_count.push_back(10000);

  EXPECT_FALSE(sobel_edge_detection.pre_processing(&task_data));

  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, NullPointerOutputData) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(new uint8_t[10000]{0});
  task_data.outputs.push_back(nullptr);
  task_data.inputs_count.push_back(10000);
  task_data.outputs_count.push_back(10000);

  EXPECT_FALSE(sobel_edge_detection.pre_processing(&task_data));

  delete[] task_data.inputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, InvalidProcessingData) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(new uint8_t[10000]{0});
  task_data.outputs.push_back(new uint8_t[10000]{0});
  task_data.inputs_count.push_back(10000);
  task_data.outputs_count.push_back(10000);
  task_data.width = 100;
  task_data.height = 100;

  std::fill(task_data.inputs[0], task_data.inputs[0] + 10000, 255);

  EXPECT_FALSE(sobel_edge_detection.run());

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}
