#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <chrono>
#include <random>
#include <vector>

#include "mpi/shuravina_o_contrast/include/ops_mpi.hpp"

TEST(shuravina_o_contrast_perf, Test_Contrast_Enhancement_Large_Image) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    auto taskDataPar = std::make_shared<ppc::core::TaskData>();

    const int input_size = 1000000;
    std::vector<uint8_t> input(input_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (int i = 0; i < input_size; ++i) {
      input[i] = dis(gen);
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());

    std::vector<uint8_t> output(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());

    shuravina_o_contrast::ContrastTaskParallel contrastTaskParallel(taskDataPar);
    ASSERT_TRUE(contrastTaskParallel.validation());

    const auto t0 = std::chrono::high_resolution_clock::now();
    contrastTaskParallel.pre_processing();
    contrastTaskParallel.run();
    contrastTaskParallel.post_processing();
    const auto t1 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Performance Test (Large Image): " << duration << " ms" << std::endl;
  }
}