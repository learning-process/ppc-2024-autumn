// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <iostream>
#include <random>

#include "mpi/zolotareva_a_smoothing_image/include/ops_mpi.hpp"
using namespace std;
std::vector<uint8_t> generateRandomImage(int height, int width, uint8_t min_value = 0, uint8_t max_value = 255) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(min_value, max_value);

  int size = height * width;
  std::vector<uint8_t> image(size);

  for (int i = 0; i < size; ++i) {
    image[i] = dis(gen);
  }

  return image;
}

void form(int height, int width) {
  boost::mpi::communicator world;
  std::vector<uint8_t> inputImage;  // generateRandomImage(height, width);
  std::vector<uint8_t> mpi_outputImage(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    inputImage = generateRandomImage(height, width);
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        cout << "inputImage[" << i * width + j << "] = " << static_cast<int>(inputImage[i * width + j]) << ' ';
      }
      cout << endl;
    }
    taskDataPar->inputs.emplace_back(inputImage.data());
    taskDataPar->inputs_count.emplace_back(height);  // кол-во строк/высота
    taskDataPar->inputs_count.emplace_back(width);   // кол-во столбцов/ширина
    taskDataPar->outputs.emplace_back(mpi_outputImage.data());
    taskDataPar->outputs_count.emplace_back(mpi_outputImage.size());
  }

  zolotareva_a_smoothing_image_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  world.barrier();
  try {
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    world.barrier();
    testMpiTaskParallel.pre_processing();
    world.barrier();
    testMpiTaskParallel.run();
    world.barrier();
    testMpiTaskParallel.post_processing();
    world.barrier();
    std::cerr << "completed on rank: " << world.rank() << std::endl;
  } catch (std::exception& e) {
    std::cerr << "Exception during scatterv on rank: " << world.rank() << " - " << e.what() << std::endl;
  }

  if (world.rank() == 0) {
    std::vector<uint8_t> seq_mpi_outputImage(width * height);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(inputImage.data());
    taskDataSeq->inputs_count.emplace_back(height);
    taskDataSeq->inputs_count.emplace_back(width);
    taskDataSeq->outputs.emplace_back(seq_mpi_outputImage.data());
    taskDataSeq->outputs_count.emplace_back(seq_mpi_outputImage.size());

    zolotareva_a_smoothing_image_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(seq_mpi_outputImage, mpi_outputImage);
  }
}

TEST(zolotareva_a_smoothing_image_mpi, Test_image) {
  boost::mpi::communicator world;
  int height = 5;
  int width = 3;
  std::vector<uint8_t> inputImage(width * height, 0);  // generateRandomImage(height, width);
  std::vector<uint8_t> mpi_outputImage(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(inputImage.data());
    taskDataPar->inputs_count.emplace_back(height);  // кол-во строк/высота
    taskDataPar->inputs_count.emplace_back(width);   // кол-во столбцов/ширина
    taskDataPar->outputs.emplace_back(mpi_outputImage.data());
    taskDataPar->outputs_count.emplace_back(mpi_outputImage.size());
  }

  zolotareva_a_smoothing_image_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  try {
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    world.barrier();
    testMpiTaskParallel.pre_processing();
    world.barrier();
    testMpiTaskParallel.run();
    world.barrier();
    testMpiTaskParallel.post_processing();
    world.barrier();
    std::cerr << "completed on rank: " << world.rank() << std::endl;
  } catch (std::exception& e) {
    std::cerr << "Exception during scatterv on rank: " << world.rank() << " - " << e.what() << std::endl;
  }
  if (world.rank() == 0) {
    std::vector<uint8_t> seq_mpi_outputImage(width * height);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(inputImage.data());
    taskDataSeq->inputs_count.emplace_back(height);
    taskDataSeq->inputs_count.emplace_back(width);
    taskDataSeq->outputs.emplace_back(seq_mpi_outputImage.data());
    taskDataSeq->outputs_count.emplace_back(seq_mpi_outputImage.size());

    zolotareva_a_smoothing_image_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(seq_mpi_outputImage, mpi_outputImage);
  }
}
TEST(zolotareva_a_smoothing_image_mpi, Test_image_random) { form(4, 4); }
TEST(zolotareva_a_smoothing_image_mpi, Test_image_random2) { form(5, 5); }
TEST(zolotareva_a_smoothing_image_mpi, Test_image_random3) { form(5, 6); }
TEST(zolotareva_a_smoothing_image_mpi, Test_image_random4) { form(6, 5); }
