// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>
#include <random>

#include "core/perf/include/perf.hpp"
#include "mpi/kalyakina_a_average_value/include/ops_mpi.hpp"

std::vector<int> RandomVectorWithFixSum(int sum, const int& count) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> result_vector(count);
  for (int i = 0; i < count - 1; i++) {
    result_vector[i] = gen() % (std::min(sum, 255) - 1);
    sum -= result_vector[i];
  }
  result_vector[count - 1] = sum;
  return result_vector;
}

TEST(kalyakina_a_average_value_mpi, Avg_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> in{};
  std::vector<double> out_mpi(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count = 120;
  int sum = 25000;
  if (world.rank() == 0) {
    in = RandomVectorWithFixSum(sum, count);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_mpi.data()));
    taskDataPar->outputs_count.emplace_back(out_mpi.size());
  }

  auto AvgMpiTaskParallel = std::make_shared<kalyakina_a_average_value_mpi::FindingAverageMPITaskParallel>(taskDataPar);
  ASSERT_EQ(AvgMpiTaskParallel->validation(), true);
  AvgMpiTaskParallel->pre_processing();
  AvgMpiTaskParallel->run();
  AvgMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(AvgMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_DOUBLE_EQ(out_mpi[0], (double)sum / count);
  }
}

TEST(kalyakina_a_average_value_mpi, Avg_task_run) {
  boost::mpi::communicator world;
  std::vector<int> in{};
  std::vector<double> out_mpi(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count = 120;
  int sum = 25000;
  if (world.rank() == 0) {
    in = RandomVectorWithFixSum(sum, count);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_mpi.data()));
    taskDataPar->outputs_count.emplace_back(out_mpi.size());
  }

  auto AvgMpiTaskParallel = std::make_shared<kalyakina_a_average_value_mpi::FindingAverageMPITaskParallel>(taskDataPar);
  ASSERT_EQ(AvgMpiTaskParallel->validation(), true);
  AvgMpiTaskParallel->pre_processing();
  AvgMpiTaskParallel->run();
  AvgMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(AvgMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_DOUBLE_EQ(out_mpi[0], (double)sum / count);
  }
}
