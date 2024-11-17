#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/vavilov_v_contrast_enhancement/include/ops_mpi.hpp"

TEST(vavilov_v_contrast_enhancement_mpi, test_task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  size_t data_size = 100000;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input(data_size, 128);
  std::vector<int> output(input.size());

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs_count.emplace_back(input.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  }

  auto testMpiTaskParallel = std::make_shared<vavilov_v_contrast_enhancement_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (size_t i = 0; i < data_size; ++i) {
      ASSERT_EQ(output[i], 0);
    }
  }
}

TEST(vavilov_v_contrast_enhancement_mpi, test_pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  size_t data_size = 100000;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input(data_size, 128);
  std::vector<int> output(input.size());

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs_count.emplace_back(input.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  }

  auto testMpiTaskParallel = std::make_shared<vavilov_v_contrast_enhancement_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (size_t i = 0; i < data_size; ++i) {
      ASSERT_EQ(output[i], 0);
    }
  }
}
