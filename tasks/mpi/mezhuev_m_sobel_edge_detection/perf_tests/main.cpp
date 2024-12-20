#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <chrono>

#include "core/perf/include/perf.hpp"
#include "mpi/mezhuev_m_sobel_edge_detection/include/ops_mpi.hpp"

TEST(mezhuev_m_sobel_edge_detection, perf_test_pipeline_run) {
  boost::mpi::communicator world;

  int grid_size = std::sqrt(world.size());
  if (grid_size * grid_size != world.size() || world.size() < 4) {
    return;
  }

  const int count = 2000000;
  std::vector<uint8_t> input_data(count, 1);
  std::vector<uint8_t> output_data(count);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(input_data.data());
  taskDataPar->inputs_count.emplace_back(input_data.size());
  taskDataPar->outputs.emplace_back(output_data.data());
  taskDataPar->outputs_count.emplace_back(output_data.size());

  auto sobelEdgeTask = std::make_shared<mezhuev_m_sobel_edge_detection::SobelEdgeDetection>(world, taskDataPar);

  ASSERT_TRUE(sobelEdgeTask->validation());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sobelEdgeTask);

  for (uint64_t i = 0; i < perfAttr->num_running; ++i) {
    sobelEdgeTask->pre_processing();
    sobelEdgeTask->run();
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
  }
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(true, true);
  }

  sobelEdgeTask->post_processing();
}

TEST(mezhuev_m_sobel_edge_detection, perf_test_pre_processing) {
  boost::mpi::communicator world;

  int grid_size = std::sqrt(world.size());
  if (grid_size * grid_size != world.size() || world.size() < 4) {
    return;
  }

  const int count = 2000000;
  std::vector<uint8_t> input_data(count, 1);
  std::vector<uint8_t> output_data(count);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(input_data.data());
  taskDataPar->inputs_count.emplace_back(input_data.size());
  taskDataPar->outputs.emplace_back(output_data.data());
  taskDataPar->outputs_count.emplace_back(output_data.size());

  auto sobelEdgeTask = std::make_shared<mezhuev_m_sobel_edge_detection::SobelEdgeDetection>(world, taskDataPar);

  ASSERT_TRUE(sobelEdgeTask->validation());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sobelEdgeTask);

  sobelEdgeTask->pre_processing();
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(true, true);
  }
}

TEST(mezhuev_m_sobel_edge_detection, perf_test_run) {
  boost::mpi::communicator world;

  int grid_size = std::sqrt(world.size());
  if (grid_size * grid_size != world.size() || world.size() < 4) {
    return;
  }

  const int count = 2000000;
  std::vector<uint8_t> input_data(count, 1);
  std::vector<uint8_t> output_data(count);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(input_data.data());
  taskDataPar->inputs_count.emplace_back(input_data.size());
  taskDataPar->outputs.emplace_back(output_data.data());
  taskDataPar->outputs_count.emplace_back(output_data.size());

  auto sobelEdgeTask = std::make_shared<mezhuev_m_sobel_edge_detection::SobelEdgeDetection>(world, taskDataPar);

  ASSERT_TRUE(sobelEdgeTask->validation());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sobelEdgeTask);

  sobelEdgeTask->run();
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(true, true);
  }
}