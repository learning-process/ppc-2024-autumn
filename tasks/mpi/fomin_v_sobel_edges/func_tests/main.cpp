#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/fomin_v_sobel_edges/include/ops_mpi.hpp"

TEST(fomin_v_sobel_edges, KnownValueTest) {
  boost::mpi::communicator world;

  // Создаем тестовое изображение 4x4
  const int width = 4;
  const int height = 4;
  std::vector<unsigned char> input_image = {100, 100, 100, 100, 100, 200, 200, 100,
                                            100, 200, 200, 100, 100, 100, 100, 100};

  // Ожидаемый результат после применения Sobel edge detection
  std::vector<unsigned char> expected_output = {0, 0, 0, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 0, 0, 0};

  // Подготавливаем TaskData для параллельной версии
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.push_back(width);
    taskDataPar->inputs_count.push_back(height);
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(input_image.data()));
    taskDataPar->outputs_count.push_back(width);
    taskDataPar->outputs_count.push_back(height);
    std::vector<unsigned char> output_image(width * height, 0);
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(output_image.data()));
  } else {
    taskDataPar->inputs_count.resize(2);
    taskDataPar->outputs_count.resize(2);
    taskDataPar->inputs.resize(1, nullptr);
    taskDataPar->outputs.resize(1, nullptr);
  }

  fomin_v_sobel_edges::SobelEdgeDetectionMPI sobelEdgeDetectionMPI(taskDataPar);
  ASSERT_TRUE(sobelEdgeDetectionMPI.validation());
  sobelEdgeDetectionMPI.pre_processing();
  sobelEdgeDetectionMPI.run();
  sobelEdgeDetectionMPI.post_processing();

  if (world.rank() == 0) {
    std::vector<unsigned char> output_image(taskDataPar->outputs[0], taskDataPar->outputs[0] + width * height);

    for (size_t i = 0; i < expected_output.size(); ++i) {
      EXPECT_EQ(expected_output[i], output_image[i]) << "Mismatch at index " << i;
    }
  }
}

TEST(fomin_v_sobel_edges, KnownValueTest_LargerImage) {
  boost::mpi::communicator world;

  // Создаем тестовое изображение 6x6
  const int width = 6;
  const int height = 6;
  std::vector<unsigned char> input_image = {100, 100, 100, 100, 100, 100, 100, 200, 200, 200, 200, 100,
                                            100, 200, 255, 255, 200, 100, 100, 200, 255, 255, 200, 100,
                                            100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 100, 100};

  // Ожидаемый результат
  std::vector<unsigned char> expected_output = {0, 0,   0,   0,   0,   0, 0, 255, 255, 255, 255, 0,
                                                0, 255, 255, 255, 255, 0, 0, 255, 255, 255, 255, 0,
                                                0, 255, 255, 255, 255, 0, 0, 0,   0,   0,   0,   0};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.push_back(width);
    taskDataPar->inputs_count.push_back(height);
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(input_image.data()));
    taskDataPar->outputs_count.push_back(width);
    taskDataPar->outputs_count.push_back(height);
    std::vector<unsigned char> output_image(width * height, 0);
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(output_image.data()));
  } else {
    taskDataPar->inputs_count.resize(2);
    taskDataPar->outputs_count.resize(2);
    taskDataPar->inputs.resize(1, nullptr);
    taskDataPar->outputs.resize(1, nullptr);
  }

  fomin_v_sobel_edges::SobelEdgeDetectionMPI sobelEdgeDetectionMPI(taskDataPar);
  ASSERT_TRUE(sobelEdgeDetectionMPI.validation());
  sobelEdgeDetectionMPI.pre_processing();
  sobelEdgeDetectionMPI.run();
  sobelEdgeDetectionMPI.post_processing();

  if (world.rank() == 0) {
    std::vector<unsigned char> output_image = *reinterpret_cast<std::vector<unsigned char>*>(taskDataPar->outputs[0]);

    for (size_t i = 0; i < expected_output.size(); ++i) {
      EXPECT_EQ(expected_output[i], output_image[i]) << "Mismatch at index " << i;
    }
  }
}

TEST(fomin_v_sobel_edges, Test_Sobel_Edge_Detection_Empty_Image) {
  boost::mpi::communicator world;
  std::vector<unsigned char> global_image;
  std::vector<unsigned char> global_output_image;

  const int width = 0;
  const int height = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.push_back(width);
    taskDataPar->inputs_count.push_back(height);
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(global_image.data()));
    taskDataPar->outputs_count.push_back(width);
    taskDataPar->outputs_count.push_back(height);
    global_output_image.resize(width * height, 0);
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(global_output_image.data()));
  }

  fomin_v_sobel_edges::SobelEdgeDetectionMPI sobelEdgeDetectionMPI(taskDataPar);
  EXPECT_FALSE(sobelEdgeDetectionMPI.validation());
}