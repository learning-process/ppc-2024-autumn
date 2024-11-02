#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <limits>

#include "core/perf/include/perf.hpp"
#include "mpi/anufriev_d_max_of_vector_elements/include/ops_mpi_anufriev.hpp"

TEST(anufriev_d_max_of_vector_elements_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int32_t> input_vector;
  int32_t result_parallel = std::numeric_limits<int32_t>::min();
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int vector_size = 50000000;

  if (world.rank() == 0) {
    input_vector.resize(vector_size, 1);
    input_vector[vector_size / 2] = 10;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    taskDataPar->inputs_count.emplace_back(input_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result_parallel));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel = std::make_shared<anufriev_d_max_of_vector_elements_parallel::VectorMaxPar>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);

  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(10, result_parallel);
  }
}

TEST(anufriev_d_max_of_vector_elements_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int32_t> input_vector;
  int32_t result_parallel = std::numeric_limits<int32_t>::min();
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int vector_size = DATA_SIZE;

  if (world.rank() == 0) {
    input_vector.resize(vector_size, 1);
    input_vector[0] = -5;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    taskDataPar->inputs_count.emplace_back(input_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result_parallel));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel = std::make_shared<anufriev_d_max_of_vector_elements_parallel::VectorMaxPar>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);

  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(1, result_parallel);
  }
}