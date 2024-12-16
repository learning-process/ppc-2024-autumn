#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <chrono>

#include "mpi/mezhuev_m_sobel_edge_detection/include/ops_mpi.hpp"

TEST(SobelEdgeDetectionMPITest, TestPrepareAndCleanupTaskData) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  mezhuev_m_sobel_edge_detection::TaskData task_data;

  task_data.width = 256;
  task_data.height = 256;
  task_data.inputs_count.push_back(task_data.width * task_data.height);
  task_data.outputs_count.push_back(task_data.width * task_data.height);
  task_data.inputs.push_back(new uint8_t[task_data.width * task_data.height]());
  task_data.outputs.push_back(new uint8_t[task_data.width * task_data.height]());

  EXPECT_EQ(static_cast<int>(task_data.width), 256);
  EXPECT_EQ(static_cast<int>(task_data.height), 256);
  EXPECT_EQ(task_data.inputs_count.size(), static_cast<size_t>(1));
  EXPECT_EQ(task_data.outputs_count.size(), static_cast<size_t>(1));
  EXPECT_EQ(task_data.inputs.size(), static_cast<size_t>(1));
  EXPECT_EQ(task_data.outputs.size(), static_cast<size_t>(1));

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];

  EXPECT_EQ(task_data.inputs[0], nullptr);
  EXPECT_EQ(task_data.outputs[0], nullptr);
}

TEST(mezhuev_m_sobel_edge_detection, TestSobelEdgeDetection) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  mezhuev_m_sobel_edge_detection::TaskData task_data;

  task_data.width = 256;
  task_data.height = 256;
  task_data.inputs_count.push_back(task_data.width * task_data.height);
  task_data.outputs_count.push_back(task_data.width * task_data.height);
  task_data.inputs.push_back(new uint8_t[task_data.width * task_data.height]());
  task_data.outputs.push_back(new uint8_t[task_data.width * task_data.height]());

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel(world);

  EXPECT_TRUE(sobel.pre_processing(&task_data));

  EXPECT_TRUE(sobel.run());

  EXPECT_TRUE(sobel.post_processing());

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}