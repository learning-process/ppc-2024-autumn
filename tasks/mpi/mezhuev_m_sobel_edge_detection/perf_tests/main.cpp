#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <chrono>

#include "core/perf/include/perf.hpp"
#include "mpi/mezhuev_m_sobel_edge_detection/include/ops_mpi.hpp"

TEST(mezhuev_m_sobel_edge_detection, TestPerformanceSmallImage) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  int width = 64;
  int height = 64;

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.width = width;
  task_data.height = height;
  task_data.inputs_count.push_back(width * height);
  task_data.outputs_count.push_back(width * height);

  task_data.inputs.push_back(new uint8_t[width * height]());
  task_data.outputs.push_back(new uint8_t[width * height]());

  ASSERT_NE(task_data.inputs[0], nullptr) << "Input data was not properly initialized!";
  ASSERT_NE(task_data.outputs[0], nullptr) << "Output data was not properly initialized!";

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel(world);

  ASSERT_TRUE(sobel.setTaskData(&task_data)) << "setTaskData failed!";
  ASSERT_TRUE(sobel.run()) << "Run failed!";

  auto start = std::chrono::high_resolution_clock::now();

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
}

TEST(mezhuev_m_sobel_edge_detection, TestPerformanceMediumImage) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  int width = 256;
  int height = 256;

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.width = width;
  task_data.height = height;
  task_data.inputs_count.push_back(width * height);
  task_data.outputs_count.push_back(width * height);

  task_data.inputs.push_back(new uint8_t[width * height]());
  task_data.outputs.push_back(new uint8_t[width * height]());

  ASSERT_NE(task_data.inputs[0], nullptr) << "Input data was not properly initialized!";
  ASSERT_NE(task_data.outputs[0], nullptr) << "Output data was not properly initialized!";

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel(world);

  ASSERT_TRUE(sobel.setTaskData(&task_data)) << "setTaskData failed!";
  ASSERT_TRUE(sobel.run()) << "Run failed!";

  auto start = std::chrono::high_resolution_clock::now();

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
}

TEST(mezhuev_m_sobel_edge_detection, TestPerformanceLargeImage) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  int width = 512;
  int height = 512;

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.width = width;
  task_data.height = height;
  task_data.inputs_count.push_back(width * height);
  task_data.outputs_count.push_back(width * height);

  task_data.inputs.push_back(new uint8_t[width * height]());
  task_data.outputs.push_back(new uint8_t[width * height]());

  ASSERT_NE(task_data.inputs[0], nullptr) << "Input data was not properly initialized!";
  ASSERT_NE(task_data.outputs[0], nullptr) << "Output data was not properly initialized!";

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel(world);

  ASSERT_TRUE(sobel.setTaskData(&task_data)) << "setTaskData failed!";
  ASSERT_TRUE(sobel.run()) << "Run failed!";

  auto start = std::chrono::high_resolution_clock::now();

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
}

TEST(mezhuev_m_sobel_edge_detection, TestPerformanceExtraLargeImage) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  int width = 1024;
  int height = 1024;

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.width = width;
  task_data.height = height;
  task_data.inputs_count.push_back(width * height);
  task_data.outputs_count.push_back(width * height);

  task_data.inputs.push_back(new uint8_t[width * height]());
  task_data.outputs.push_back(new uint8_t[width * height]());

  ASSERT_NE(task_data.inputs[0], nullptr) << "Input data was not properly initialized!";
  ASSERT_NE(task_data.outputs[0], nullptr) << "Output data was not properly initialized!";

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel(world);

  ASSERT_TRUE(sobel.setTaskData(&task_data)) << "setTaskData failed!";
  ASSERT_TRUE(sobel.run()) << "Run failed!";

  auto start = std::chrono::high_resolution_clock::now();

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
}