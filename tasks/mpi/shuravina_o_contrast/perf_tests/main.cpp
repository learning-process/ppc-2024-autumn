#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <chrono>
#include <random>
#include <vector>

#include "mpi/shuravina_o_contrast/include/ops_mpi.hpp"

namespace shuravina_o_contrast {

std::vector<uint8_t> generateRandomImage(size_t size) {
  std::vector<uint8_t> image(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, 255);

  for (size_t i = 0; i < size; ++i) {
    image[i] = static_cast<uint8_t>(distrib(gen));
  }

  return image;
}

}  // namespace shuravina_o_contrast

TEST(shuravina_o_contrast_perf, Test_Contrast_Enhancement_Small_Image) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    auto taskDataPar = std::make_shared<ppc::core::TaskData>();

    std::vector<uint8_t> input = shuravina_o_contrast::generateRandomImage(1000);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());

    std::vector<uint8_t> output(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());

    auto start = std::chrono::high_resolution_clock::now();

    shuravina_o_contrast::ContrastTaskParallel contrastTaskParallel(taskDataPar);
    ASSERT_TRUE(contrastTaskParallel.pre_processing());
    ASSERT_TRUE(contrastTaskParallel.run());
    ASSERT_TRUE(contrastTaskParallel.post_processing());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Time taken for small image: " << elapsed.count() << " seconds" << std::endl;
  }
}

TEST(shuravina_o_contrast_perf, Test_Contrast_Enhancement_Large_Image) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    auto taskDataPar = std::make_shared<ppc::core::TaskData>();

    std::vector<uint8_t> input = shuravina_o_contrast::generateRandomImage(1000000);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());

    std::vector<uint8_t> output(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());

    auto start = std::chrono::high_resolution_clock::now();

    shuravina_o_contrast::ContrastTaskParallel contrastTaskParallel(taskDataPar);
    ASSERT_TRUE(contrastTaskParallel.pre_processing());
    ASSERT_TRUE(contrastTaskParallel.run());
    ASSERT_TRUE(contrastTaskParallel.post_processing());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Time taken for large image: " << elapsed.count() << " seconds" << std::endl;
  }
}