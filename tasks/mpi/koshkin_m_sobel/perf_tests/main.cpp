#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "../include/ops_mpi.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"

static std::vector<uint8_t> make_img(size_t width, size_t height) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distrib(0, 255);
  std::vector<uint8_t> vec(width * height);
  for (size_t i = 0; i < width * height; i++) {
    vec[i] = distrib(gen);
  }
  return vec;
}

TEST(koshkin_m_sobel_mpi_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;

  const auto width = 1200;
  const auto height = 1200;
  std::vector<uint8_t> in = make_img(width, height);
  std::vector<uint8_t> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs = {reinterpret_cast<uint8_t *>(in.data())};
    taskDataPar->inputs_count = {width, height};
    taskDataPar->outputs = {reinterpret_cast<uint8_t *>(out.data())};
    taskDataPar->outputs_count = {width, height};
  }

  // Create Task
  auto testTaskParallel = std::make_shared<koshkin_m_sobel_mpi::TestTaskParallel>(taskDataPar);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(koshkin_m_sobel_mpi_perf_test, test_task_run) {
  boost::mpi::communicator world;

  const auto width = 1200;
  const auto height = 1200;
  std::vector<uint8_t> in = make_img(width, height);
  std::vector<uint8_t> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs = {reinterpret_cast<uint8_t *>(in.data())};
    taskDataPar->inputs_count = {width, height};
    taskDataPar->outputs = {reinterpret_cast<uint8_t *>(out.data())};
    taskDataPar->outputs_count = {width, height};
  }

  // Create Task
  auto testTaskParallel = std::make_shared<koshkin_m_sobel_mpi::TestTaskParallel>(taskDataPar);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}
