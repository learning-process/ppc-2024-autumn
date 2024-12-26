#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/fomin_v_sobel_edges/include/ops_mpi.hpp"

TEST(fomin_v_sobel_edges, Test_Sobel_Edge_Detection) {
  boost::mpi::communicator world;
  std::vector<unsigned char> global_image;
  std::vector<unsigned char> global_output_image;

  // Create a test 4x4 image
  const int width = 4;
  const int height = 4;
  global_image.resize(width * height, 100);
  for (int i = 1; i < height - 1; ++i) {
    for (int j = 1; j < width - 1; ++j) {
      global_image[i * width + j] = 200;
    }
  }

  // Prepare TaskData for parallel version
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_image.data()));
    taskDataPar->inputs_count.emplace_back(global_image.size());
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->inputs_count.emplace_back(width);
    global_output_image.resize(width * height, 0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_output_image.data()));
    taskDataPar->outputs_count.emplace_back(global_output_image.size());
  }

  // Execute parallel Sobel edge detection
  fomin_v_sobel_edges::SobelEdgeDetectionMPI sobelEdgeDetectionMPI(taskDataPar);
  ASSERT_EQ(sobelEdgeDetectionMPI.validation(), true);
  sobelEdgeDetectionMPI.pre_processing();
  sobelEdgeDetectionMPI.run();
  sobelEdgeDetectionMPI.post_processing();

  if (world.rank() == 0) {
    // Prepare TaskData for sequential version
    std::vector<unsigned char> reference_output_image(width * height, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_image.data()));
    taskDataSeq->inputs_count.emplace_back(global_image.size());
    taskDataSeq->inputs_count.emplace_back(height);
    taskDataSeq->inputs_count.emplace_back(width);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_output_image.data()));
    taskDataSeq->outputs_count.emplace_back(reference_output_image.size());

    // Execute sequential Sobel edge detection
    fomin_v_sobel_edges::SobelEdgeDetection sobelEdgeDetection(taskDataSeq);
    ASSERT_EQ(sobelEdgeDetection.validation(), true);
    sobelEdgeDetection.pre_processing();
    sobelEdgeDetection.run();
    sobelEdgeDetection.post_processing();

    // Compare outputs
    for (size_t i = 0; i < reference_output_image.size(); ++i) {
      EXPECT_EQ(reference_output_image[i], global_output_image[i]);
    }
  }
}

TEST(fomin_v_sobel_edges, Test_Sobel_Edge_Detection_Large_Image) {
  boost::mpi::communicator world;
  std::vector<unsigned char> global_image;

  // Create a larger 8x8 test image
  const int width = 8;
  const int height = 8;
  global_image.resize(width * height, 100);
  for (int i = 2; i < 6; ++i) {
    for (int j = 2; j < 6; ++j) {
      global_image[i * width + j] = 200;
    }
  }
  std::vector<unsigned char> global_output_image(global_image.size(), 0);

  // Prepare TaskData for parallel version
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_image.data()));
    taskDataPar->inputs_count.emplace_back(global_image.size());
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_output_image.data()));
    taskDataPar->outputs_count.emplace_back(global_output_image.size());
  }

  // Execute parallel Sobel edge detection
  fomin_v_sobel_edges::SobelEdgeDetectionMPI sobelEdgeDetectionMPI(taskDataPar);
  ASSERT_EQ(sobelEdgeDetectionMPI.validation(), true);
  sobelEdgeDetectionMPI.pre_processing();
  sobelEdgeDetectionMPI.run();
  sobelEdgeDetectionMPI.post_processing();

  if (world.rank() == 0) {
    // Prepare TaskData for sequential version
    std::vector<unsigned char> reference_output_image(width * height, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_image.data()));
    taskDataSeq->inputs_count.emplace_back(global_image.size());
    taskDataSeq->inputs_count.emplace_back(height);
    taskDataSeq->inputs_count.emplace_back(width);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_output_image.data()));
    taskDataSeq->outputs_count.emplace_back(reference_output_image.size());

    // Execute sequential Sobel edge detection
    fomin_v_sobel_edges::SobelEdgeDetection sobelEdgeDetection(taskDataSeq);
    ASSERT_EQ(sobelEdgeDetection.validation(), true);
    sobelEdgeDetection.pre_processing();
    sobelEdgeDetection.run();
    sobelEdgeDetection.post_processing();

    // Compare outputs
    for (size_t i = 0; i < reference_output_image.size(); ++i) {
      EXPECT_EQ(reference_output_image[i], global_output_image[i]);
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
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_image.data()));
    taskDataPar->inputs_count.emplace_back(global_image.size());
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_output_image.data()));
    taskDataPar->outputs_count.emplace_back(global_output_image.size());
  }

  fomin_v_sobel_edges::SobelEdgeDetectionMPI sobelEdgeDetectionMPI(taskDataPar);
  EXPECT_FALSE(sobelEdgeDetectionMPI.validation());
}