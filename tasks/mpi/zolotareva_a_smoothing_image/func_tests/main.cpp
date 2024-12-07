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
  std::vector<uint8_t> inputImage;
  std::vector<uint8_t> mpi_outputImage(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    inputImage = generateRandomImage(height, width);
    taskDataPar->inputs.emplace_back(inputImage.data());
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->outputs.emplace_back(mpi_outputImage.data());
    taskDataPar->outputs_count.emplace_back(mpi_outputImage.size());
  }

  zolotareva_a_smoothing_image_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  world.barrier();
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  world.barrier();

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

TEST(zolotareva_a_smoothing_image_mpi, Test_image_with_nulls) {
  boost::mpi::communicator world;
  int height = 5;
  int width = 3;
  std::vector<uint8_t> inputImage(width * height, 0);
  std::vector<uint8_t> mpi_outputImage(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(inputImage.data());
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->outputs.emplace_back(mpi_outputImage.data());
    taskDataPar->outputs_count.emplace_back(mpi_outputImage.size());
  }

  zolotareva_a_smoothing_image_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

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
TEST(zolotareva_a_smoothing_image_mpi, Test_image_random_100X100) { form(100, 100); }
TEST(zolotareva_a_smoothing_image_mpi, Test_image_random_100X500) { form(100, 500); }
TEST(zolotareva_a_smoothing_image_mpi, Test_image_random_500X230) { form(500, 230); }
TEST(zolotareva_a_smoothing_image_mpi, Test_image_random_500X500) { form(500, 500); }
TEST(zolotareva_a_smoothing_image_mpi, Test_image_random_1000X1000) { form(10001, 10000); }
