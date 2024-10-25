// Filatev Vladislav Sum_of_matrix_elements
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/filatev_v_sum_of_matrix_elements/include/ops_mpi.hpp"

TEST(filatev_v_sum_of_matrix_elements_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  const int count = 10000;
  std::vector<int> out;
  std::vector<std::vector<int>> in;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<std::vector<int>>(count, std::vector<int>(count, 1));
    out = std::vector<int>(1, 0);
    for (int i = 0; i < count; i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(count);
    taskDataPar->inputs_count.emplace_back(count);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto sumMatrixparallel =
      std::make_shared<filatev_v_sum_of_matrix_elements_mpi::SumMatrixParallel>(taskDataPar, world);
  ASSERT_EQ(sumMatrixparallel->validation(), true);
  sumMatrixparallel->pre_processing();
  sumMatrixparallel->run();
  sumMatrixparallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sumMatrixparallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(100000000, out[0]);
  }
}

TEST(filatev_v_sum_of_matrix_elements_mpi, test_task_run) {
  boost::mpi::communicator world;
  const int count = 10000;
  std::vector<int> out;
  std::vector<std::vector<int>> in;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<std::vector<int>>(count, std::vector<int>(count, 1));
    out = std::vector<int>(1, 0);
    for (int i = 0; i < count; i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(count);
    taskDataPar->inputs_count.emplace_back(count);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto sumMatrixparallel = 
      std::make_shared<filatev_v_sum_of_matrix_elements_mpi::SumMatrixParallel>(taskDataPar, world);
  ASSERT_EQ(sumMatrixparallel->validation(), true);
  sumMatrixparallel->pre_processing();
  sumMatrixparallel->run();
  sumMatrixparallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sumMatrixparallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(100000000, out[0]);
  }
}
