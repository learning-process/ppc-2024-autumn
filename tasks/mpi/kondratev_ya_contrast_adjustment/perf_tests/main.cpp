// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kondratev_ya_contrast_adjustment/include/ops_mpi.hpp"

namespace kondratev_ya_contrast_adjustment_mpi {
std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> genRandomData(uint32_t size) {
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> buff(size);
  std::random_device dev;
  std::mt19937 gen(dev());
  for (uint32_t i = 0; i < size; i++) {
    buff[i] = gen() % 256;
  }
  return buff;
}
}  // namespace kondratev_ya_contrast_adjustment_mpi

TEST(kondratev_ya_contrast_adjustment_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<double> contrast;

  if (world.rank() == 0) {
    int size = 800000;
    input = kondratev_ya_contrast_adjustment_mpi::genRandomData(size);

    res.resize(input.size());
    contrast = std::make_shared<double>(1.25);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  auto testMpiTaskParallel = std::make_shared<kondratev_ya_contrast_adjustment_mpi::TestMPITaskParallel>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(kondratev_ya_contrast_adjustment_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<double> contrast;

  if (world.rank() == 0) {
    int size = 800000;
    input = kondratev_ya_contrast_adjustment_mpi::genRandomData(size);
    res.resize(input.size());
    contrast = std::make_shared<double>(1.25);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  auto testMpiTaskParallel = std::make_shared<kondratev_ya_contrast_adjustment_mpi::TestMPITaskParallel>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}
