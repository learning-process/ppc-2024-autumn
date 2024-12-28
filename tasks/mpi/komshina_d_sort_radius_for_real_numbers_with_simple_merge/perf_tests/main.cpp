#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_mpi.hpp"

namespace mpi = boost::mpi;

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, test_pipeline_run) {
  mpi::communicator world;
  int count = 10;
  std::vector<double> inputData;
  if (world.rank() == 0) {
    inputData = {3.14, 2.71, 1.41, 1.73, 0.6, 1.61, 2.0, 2.236, 3.01, 4.0};
  }

  std::vector<double> resPar(count, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataPar->inputs_count.emplace_back(count);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resPar.data()));
    taskDataPar->outputs_count.emplace_back(count);
  }

  auto testMpiTaskParallel =
      std::make_shared<komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, test_task_run) {
  mpi::communicator world;
  int count = 10;
  std::vector<double> inputData;
  if (world.rank() == 0) {
    inputData = {3.14, 2.71, 1.41, 1.73, 0.6, 1.61, 2.0, 2.236, 3.01, 4.0};
  }

  std::vector<double> resPar(count, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataPar->inputs_count.emplace_back(count);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resPar.data()));
    taskDataPar->outputs_count.emplace_back(count);
  }

  auto testMpiTaskParallel =
      std::make_shared<komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}