#include <gtest/gtest.h>

#include <boost/mpi.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/milovankin_m_sum_of_vector_elements/include/ops_mpi.hpp"

TEST(milovankin_m_sum_of_vector_elements_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int32_t> input_vector;
  std::vector<int64_t> result_parallel(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int vector_size = 50'000'000;

  if (world.rank() == 0) {
    input_vector.resize(vector_size, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    taskDataPar->inputs_count.emplace_back(input_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_parallel.data()));
    taskDataPar->outputs_count.emplace_back(result_parallel.size());
  }

  auto testMpiTaskParallel = std::make_shared<milovankin_m_sum_of_vector_elements_parallel::VectorSumPar>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);

  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(vector_size, result_parallel[0]);
  }
}

TEST(milovankin_m_sum_of_vector_elements_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int32_t> input_vector;
  std::vector<int64_t> result_parallel(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int vector_size = 50'000'000;

  if (world.rank() == 0) {
    input_vector = std::vector<int32_t>(vector_size, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    taskDataPar->inputs_count.emplace_back(input_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_parallel.data()));
    taskDataPar->outputs_count.emplace_back(result_parallel.size());
  }

  auto testMpiTaskParallel = std::make_shared<milovankin_m_sum_of_vector_elements_parallel::VectorSumPar>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(vector_size, result_parallel[0]);
  }
}
