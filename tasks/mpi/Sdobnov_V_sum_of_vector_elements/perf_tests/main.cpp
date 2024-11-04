#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/Sdobnov_V_sum_of_vector_elements/include/ops_mpi.hpp"

TEST(Sdobnov_V_sum_of_vector_elements_par, test_pipeline_run) {
  boost::mpi::communicator world;
  int rows = 10000;
  int columns = 10000;
  int res;
  std::vector<std::vector<int>> input = Sdobnov_V_sum_of_vector_elements::generate_random_matrix(rows, columns, 1, 1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));
  }
  auto test = std::make_shared<Sdobnov_V_sum_of_vector_elements::SumVecElemParallel>(taskDataPar);

  test->validation();
  test->pre_processing();
  test->run();
  test->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(rows * columns, res);
  }
}

TEST(Sdobnov_V_sum_of_vector_elements_par, test_task_run) {
  boost::mpi::communicator world;
  int rows = 10000;
  int columns = 10000;
  int res;
  std::vector<std::vector<int>> input = Sdobnov_V_sum_of_vector_elements::generate_random_matrix(rows, columns, 1, 1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));
  }
  auto test = std::make_shared<Sdobnov_V_sum_of_vector_elements::SumVecElemParallel>(taskDataPar);

  test->validation();
  test->pre_processing();
  test->run();
  test->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(rows * columns, res);
  }
}