#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/shuravina_o_contrast/include/ops_mpi.hpp"

std::vector<uint8_t> genRandomData(uint32_t size) {
  std::vector<uint8_t> buff(size);
  std::random_device dev;
  std::mt19937 gen(dev());
  for (uint32_t i = 0; i < size; i++) {
    buff[i] = gen() % 256;
  }
  return buff;
}

TEST(shuravina_o_contrast, Test_Contrast_Enhancement_Functional) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    {
      auto taskDataPar = std::make_shared<ppc::core::TaskData>();

      std::vector<uint8_t> input = {};
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
      taskDataPar->inputs_count.emplace_back(input.size());

      std::vector<uint8_t> output(input.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
      taskDataPar->outputs_count.emplace_back(output.size());

      shuravina_o_contrast::ContrastTaskParallel contrastTaskParallel(taskDataPar);
      ASSERT_TRUE(contrastTaskParallel.validation());
    }

    {
      auto taskDataPar = std::make_shared<ppc::core::TaskData>();

      std::vector<uint8_t> input = {50};
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
      taskDataPar->inputs_count.emplace_back(input.size());

      std::vector<uint8_t> output(input.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
      taskDataPar->outputs_count.emplace_back(output.size());

      shuravina_o_contrast::ContrastTaskParallel contrastTaskParallel(taskDataPar);
      ASSERT_TRUE(contrastTaskParallel.validation());
    }

    {
      auto taskDataPar = std::make_shared<ppc::core::TaskData>();

      std::vector<uint8_t> input = {255, 255, 255, 255, 255};
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
      taskDataPar->inputs_count.emplace_back(input.size());

      std::vector<uint8_t> output(input.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
      taskDataPar->outputs_count.emplace_back(output.size());

      shuravina_o_contrast::ContrastTaskParallel contrastTaskParallel(taskDataPar);
      ASSERT_TRUE(contrastTaskParallel.validation());
    }

    {
      auto taskDataPar = std::make_shared<ppc::core::TaskData>();

      std::vector<uint8_t> input = {0, 0, 0, 0, 0};
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
      taskDataPar->inputs_count.emplace_back(input.size());

      std::vector<uint8_t> output(input.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
      taskDataPar->outputs_count.emplace_back(output.size());

      shuravina_o_contrast::ContrastTaskParallel contrastTaskParallel(taskDataPar);
      ASSERT_TRUE(contrastTaskParallel.validation());
    }
  }
}

TEST(shuravina_o_contrast_perf, Test_Contrast_Enhancement_Performance) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    auto taskDataPar = std::make_shared<ppc::core::TaskData>();

    int size = 800000;
    std::vector<uint8_t> input = genRandomData(size);
    std::vector<uint8_t> output(input.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());

    auto contrastTaskParallel = std::make_shared<shuravina_o_contrast::ContrastTaskParallel>(taskDataPar);

    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    auto perfResults = std::make_shared<ppc::core::PerfResults>();
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(contrastTaskParallel);
    perfAnalyzer->pipeline_run(perfAttr, perfResults);

    if (world.rank() == 0) {
      ppc::core::Perf::print_perf_statistic(perfResults);
    }
  }
}