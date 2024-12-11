#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/shuravina_o_contrast/include/ops_mpi.hpp"
#include <boost/mpi/collectives.hpp>

std::vector<uint8_t> generateRandomImage(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);

  std::vector<uint8_t> input(size);
  for (size_t i = 0; i < size; ++i) {
    input[i] = static_cast<uint8_t>(dis(gen));
  }
  return input;
}

TEST(shuravina_o_contrast, Test_Contrast_Enhancement_Single_Element_Input) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<uint8_t> input;
  if (world.rank() == 0) {
    input = {50};
  }

  boost::mpi::broadcast(world, input, 0);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskDataPar->inputs_count.emplace_back(input.size());

  std::vector<uint8_t> output(input.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskDataPar->outputs_count.emplace_back(output.size());

  shuravina_o_contrast::ContrastTaskParallel contrastTaskParallel(taskDataPar);
  ASSERT_TRUE(contrastTaskParallel.validation());
  contrastTaskParallel.pre_processing();
  contrastTaskParallel.run();
  contrastTaskParallel.post_processing();

  std::vector<uint8_t> gatheredOutput;
  if (world.rank() == 0) {
    gatheredOutput.resize(input.size() * world.size());
  }
  boost::mpi::gather(world, output.data(), output.size(), gatheredOutput.data(), 0);

  if (world.rank() == 0) {
    ASSERT_EQ(gatheredOutput[0], 0);

    for (size_t i = 1; i < gatheredOutput.size(); ++i) {
      ASSERT_EQ(gatheredOutput[i], 0);
    }
  }
}

TEST(shuravina_o_contrast, Test_Contrast_Enhancement_Random_Small_Image) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    auto taskDataPar = std::make_shared<ppc::core::TaskData>();

    std::vector<uint8_t> input = generateRandomImage(100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());

    std::vector<uint8_t> output(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());

    shuravina_o_contrast::ContrastTaskParallel contrastTaskParallel(taskDataPar);
    ASSERT_TRUE(contrastTaskParallel.validation());
    contrastTaskParallel.pre_processing();
    contrastTaskParallel.run();
    contrastTaskParallel.post_processing();

    auto min_val = *std::min_element(input.begin(), input.end());
    auto max_val = *std::max_element(input.begin(), input.end());
    for (size_t i = 0; i < input.size(); ++i) {
      auto expected = static_cast<uint8_t>((input[i] - min_val) * 255.0 / (max_val - min_val));
      ASSERT_EQ(output[i], expected);
    }
  }
}

TEST(shuravina_o_contrast, Test_Contrast_Enhancement_Random_Medium_Image) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    auto taskDataPar = std::make_shared<ppc::core::TaskData>();

    std::vector<uint8_t> input = generateRandomImage(1000);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());

    std::vector<uint8_t> output(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());

    shuravina_o_contrast::ContrastTaskParallel contrastTaskParallel(taskDataPar);
    ASSERT_TRUE(contrastTaskParallel.validation());
    contrastTaskParallel.pre_processing();
    contrastTaskParallel.run();
    contrastTaskParallel.post_processing();

    auto min_val = *std::min_element(input.begin(), input.end());
    auto max_val = *std::max_element(input.begin(), input.end());
    for (size_t i = 0; i < input.size(); ++i) {
      auto expected = static_cast<uint8_t>((input[i] - min_val) * 255.0 / (max_val - min_val));
      ASSERT_EQ(output[i], expected);
    }
  }
}

TEST(shuravina_o_contrast, Test_Contrast_Enhancement_Random_Large_Image) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    auto taskDataPar = std::make_shared<ppc::core::TaskData>();

    std::vector<uint8_t> input = generateRandomImage(10000);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());

    std::vector<uint8_t> output(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());

    shuravina_o_contrast::ContrastTaskParallel contrastTaskParallel(taskDataPar);
    ASSERT_TRUE(contrastTaskParallel.validation());
    contrastTaskParallel.pre_processing();
    contrastTaskParallel.run();
    contrastTaskParallel.post_processing();

    auto min_val = *std::min_element(input.begin(), input.end());
    auto max_val = *std::max_element(input.begin(), input.end());
    for (size_t i = 0; i < input.size(); ++i) {
      auto expected = static_cast<uint8_t>((input[i] - min_val) * 255.0 / (max_val - min_val));
      ASSERT_EQ(output[i], expected);
    }
  }
}