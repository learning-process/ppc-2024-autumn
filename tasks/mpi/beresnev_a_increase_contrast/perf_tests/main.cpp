// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/beresnev_a_increase_contrast/include/ops_mpi.hpp"

TEST(beresnev_a_increase_contrast_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  double factor = 1.0;
  int width = 1000;
  int height = 5000;
  int max_color = 255;

  size_t file_size;

  std::vector<uint8_t> input_buffer;
  std::vector<uint8_t> out_buffer;
  std::vector<uint8_t> ans_buffer;

  std::string head =
      "P6\n" + std::to_string(width) + " " + std::to_string(height) + '\n' + std::to_string(max_color) + '\n';

  file_size = head.size() + width * height * 3;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<uint8_t> inp(width * height * 3, 0);

    input_buffer.reserve(file_size);
    input_buffer.insert(input_buffer.end(), head.begin(), head.end());
    input_buffer.insert(input_buffer.end(), inp.data(), inp.data() + inp.size());

    std::vector<uint8_t> ans(width * height * 3, 0);

    ans_buffer.reserve(file_size);
    ans_buffer.insert(ans_buffer.end(), head.begin(), head.end());
    ans_buffer.insert(ans_buffer.end(), ans.data(), ans.data() + ans.size());

    out_buffer = std::vector<uint8_t>(file_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_buffer.data()));
    taskDataPar->inputs_count.emplace_back(file_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&factor));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_buffer));
    taskDataPar->outputs_count.emplace_back(file_size);
  }

  auto testMpiTaskParallel = std::make_shared<beresnev_a_increase_contrast_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(ans_buffer, out_buffer);
  }
}

TEST(beresnev_a_increase_contrast_mpi, test_task_run) {
  boost::mpi::communicator world;
  double factor = 1.0;
  int width = 1000;
  int height = 5000;
  int max_color = 255;

  size_t file_size;

  std::vector<uint8_t> input_buffer;
  std::vector<uint8_t> out_buffer;
  std::vector<uint8_t> ans_buffer;

  std::string head =
      "P6\n" + std::to_string(width) + " " + std::to_string(height) + '\n' + std::to_string(max_color) + '\n';

  file_size = head.size() + width * height * 3;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<uint8_t> inp(width * height * 3, 0);

    input_buffer.reserve(file_size);
    input_buffer.insert(input_buffer.end(), head.begin(), head.end());
    input_buffer.insert(input_buffer.end(), inp.data(), inp.data() + inp.size());

    std::vector<uint8_t> ans(width * height * 3, 0);

    ans_buffer.reserve(file_size);
    ans_buffer.insert(ans_buffer.end(), head.begin(), head.end());
    ans_buffer.insert(ans_buffer.end(), ans.data(), ans.data() + ans.size());

    out_buffer = std::vector<uint8_t>(file_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_buffer.data()));
    taskDataPar->inputs_count.emplace_back(file_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&factor));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_buffer));
    taskDataPar->outputs_count.emplace_back(file_size);
  }

  auto testMpiTaskParallel = std::make_shared<beresnev_a_increase_contrast_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(ans_buffer, out_buffer);
  }
}