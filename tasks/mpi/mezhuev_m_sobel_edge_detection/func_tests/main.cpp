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

TEST(mezhuev_m_sobel_edge_detection, func_test_validation_empty_inputs_outputs) {
  boost::mpi::communicator world;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.clear();
  taskData->outputs.clear();

  auto sobelEdgeTask = std::make_shared<mezhuev_m_sobel_edge_detection::SobelEdgeDetection>(world, taskData);

  ASSERT_FALSE(sobelEdgeTask->validation());
}

TEST(mezhuev_m_sobel_edge_detection, func_test_validation_wrong_data_size) {
  boost::mpi::communicator world;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(new int[10]));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(new int[10]));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(new int[10]));

  auto sobelEdgeTask = std::make_shared<mezhuev_m_sobel_edge_detection::SobelEdgeDetection>(world, taskData);

  ASSERT_FALSE(sobelEdgeTask->validation());
}

TEST(mezhuev_m_sobel_edge_detection, func_test_validation_null_pointers) {
  boost::mpi::communicator world;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(nullptr);
  taskData->outputs.push_back(nullptr);

  auto sobelEdgeTask = std::make_shared<mezhuev_m_sobel_edge_detection::SobelEdgeDetection>(world, taskData);

  ASSERT_FALSE(sobelEdgeTask->validation());
}

TEST(mezhuev_m_sobel_edge_detection, func_test_validation_inputs_outputs_mismatch) {
  boost::mpi::communicator world;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(new int[10]));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(new int[10]));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(new int[10]));

  auto sobelEdgeTask = std::make_shared<mezhuev_m_sobel_edge_detection::SobelEdgeDetection>(world, taskData);

  ASSERT_FALSE(sobelEdgeTask->validation());
}

TEST(mezhuev_m_sobel_edge_detection, func_test_validation_valid_data) {
  boost::mpi::communicator world;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(new int[10]));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(new int[10]));
  taskData->inputs_count.push_back(10);
  taskData->outputs_count.push_back(10);

  auto sobelEdgeTask = std::make_shared<mezhuev_m_sobel_edge_detection::SobelEdgeDetection>(world, taskData);

  ASSERT_TRUE(sobelEdgeTask->validation());
}

TEST(mezhuev_m_sobel_edge_detection, func_test_pre_processing_empty_inputs) {
  boost::mpi::communicator world;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.clear();
  taskData->inputs_count.push_back(0);

  auto sobelEdgeTask = std::make_shared<mezhuev_m_sobel_edge_detection::SobelEdgeDetection>(world, taskData);
  ASSERT_FALSE(sobelEdgeTask->validation());
  ASSERT_FALSE(sobelEdgeTask->pre_processing());
}

TEST(mezhuev_m_sobel_edge_detection, func_test_pre_processing_valid_data) {
  boost::mpi::communicator world;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(new int[10]));
  taskData->inputs_count.push_back(10);

  auto sobelEdgeTask = std::make_shared<mezhuev_m_sobel_edge_detection::SobelEdgeDetection>(world, taskData);

  ASSERT_FALSE(sobelEdgeTask->validation());
  ASSERT_TRUE(sobelEdgeTask->pre_processing());

  const auto& gradient_x = sobelEdgeTask->get_gradient_x();
  const auto& gradient_y = sobelEdgeTask->get_gradient_y();

  ASSERT_EQ(static_cast<int>(gradient_x.size()), 10);
  ASSERT_EQ(static_cast<int>(gradient_y.size()), 10);
}

TEST(mezhuev_m_sobel_edge_detection, func_test_pre_processing_zero_size_data) {
  boost::mpi::communicator world;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(new int[0]));
  taskData->inputs_count.push_back(0);

  auto sobelEdgeTask = std::make_shared<mezhuev_m_sobel_edge_detection::SobelEdgeDetection>(world, taskData);
  ASSERT_FALSE(sobelEdgeTask->validation());
  ASSERT_FALSE(sobelEdgeTask->pre_processing());
}

TEST(mezhuev_m_sobel_edge_detection, func_test_pre_processing_multiple_inputs) {
  boost::mpi::communicator world;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(new int[10]));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(new int[10]));
  taskData->inputs_count.push_back(10);
  taskData->inputs_count.push_back(10);

  auto sobelEdgeTask = std::make_shared<mezhuev_m_sobel_edge_detection::SobelEdgeDetection>(world, taskData);
  ASSERT_FALSE(sobelEdgeTask->validation());
  ASSERT_FALSE(sobelEdgeTask->pre_processing());
}

TEST(mezhuev_m_sobel_edge_detection, func_test_run_invalid_inputs) {
  boost::mpi::communicator world;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(nullptr);
  taskData->outputs.push_back(nullptr);
  taskData->inputs_count.push_back(0);

  auto sobelEdgeTask = std::make_shared<mezhuev_m_sobel_edge_detection::SobelEdgeDetection>(world, taskData);
  ASSERT_FALSE(sobelEdgeTask->validation());
  ASSERT_FALSE(sobelEdgeTask->pre_processing());
  ASSERT_FALSE(sobelEdgeTask->run());
}

TEST(mezhuev_m_sobel_edge_detection, func_test_run_single_pixel_image) {
  boost::mpi::communicator world;

  size_t data_size = 1;
  auto taskData = std::make_shared<ppc::core::TaskData>();

  uint8_t input_image[1][1] = {{255}};
  uint8_t output_image[1][1] = {{0}};

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_image));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_image));
  taskData->inputs_count.push_back(data_size);

  auto sobelEdgeTask = std::make_shared<mezhuev_m_sobel_edge_detection::SobelEdgeDetection>(world, taskData);
  ASSERT_FALSE(sobelEdgeTask->validation());
  ASSERT_TRUE(sobelEdgeTask->pre_processing());
  ASSERT_FALSE(sobelEdgeTask->run());

  ASSERT_EQ(output_image[0][0], 0);
}

TEST(mezhuev_m_sobel_edge_detection, func_test_run_parallel_processing) {
  boost::mpi::communicator world;

  size_t data_size = 6;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  uint8_t input_image[6][6] = {{255, 255, 255, 255, 255, 255}, {255, 255, 255, 255, 255, 255},
                               {255, 255, 255, 255, 255, 255}, {255, 255, 255, 255, 255, 255},
                               {255, 255, 255, 255, 255, 255}, {255, 255, 255, 255, 255, 255}};
  uint8_t output_image[6][6] = {{0}};

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_image));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_image));
  taskData->inputs_count.push_back(data_size);

  auto sobelEdgeTask = std::make_shared<mezhuev_m_sobel_edge_detection::SobelEdgeDetection>(world, taskData);
  ASSERT_FALSE(sobelEdgeTask->validation());
  ASSERT_TRUE(sobelEdgeTask->pre_processing());
  ASSERT_TRUE(sobelEdgeTask->run());

  for (size_t i = 0; i < data_size; ++i) {
    for (size_t j = 0; j < data_size; ++j) {
      ASSERT_GE(output_image[i][j], 0);
    }
  }
}

TEST(mezhuev_m_sobel_edge_detection, func_test_run_synchronization_between_processes) {
  boost::mpi::communicator world;

  size_t data_size = 6;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  uint8_t input_image[6][6] = {{255, 255, 255, 255, 255, 255}, {255, 255, 255, 255, 255, 255},
                               {255, 255, 255, 255, 255, 255}, {255, 255, 255, 255, 255, 255},
                               {255, 255, 255, 255, 255, 255}, {255, 255, 255, 255, 255, 255}};
  uint8_t output_image[6][6] = {{0}};

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_image));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_image));
  taskData->inputs_count.push_back(data_size);

  auto sobelEdgeTask = std::make_shared<mezhuev_m_sobel_edge_detection::SobelEdgeDetection>(world, taskData);
  ASSERT_FALSE(sobelEdgeTask->validation());
  ASSERT_TRUE(sobelEdgeTask->pre_processing());
  ASSERT_TRUE(sobelEdgeTask->run());

  for (size_t i = 0; i < data_size; ++i) {
    for (size_t j = 0; j < data_size; ++j) {
      ASSERT_GE(output_image[i][j], 0);
    }
  }
}