// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/zaitsev_a_min_of_vector_elements/include/ops_mpi.hpp"

namespace zaitsev_a_min_of_vector_elements_mpi {
std::vector<int> getRandomVector(int sz, int minRangeValue, int maxRangeValue) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = minRangeValue + gen() % (maxRangeValue - minRangeValue + 1);
  }
  return vec;
}
}  // namespace zaitsev_a_min_of_vector_elements_mpi

TEST(zaitsev_a_min_of_vector_elements_mpi, test_pipeline_run) {
  const int extrema = -1000;
  const int minRangeValue = -500;
  const int maxRangeValue = 500;

  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_min(1, maxRangeValue + 1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 10e6;
    global_vec = zaitsev_a_min_of_vector_elements_mpi::getRandomVector(count_size_vector, minRangeValue, maxRangeValue);
    global_vec[global_vec.size() / 2] = extrema;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  auto minOfVectorElementsParallel =
      std::make_shared<zaitsev_a_min_of_vector_elements_mpi::MinOfVectorElementsParallel>(taskDataPar);
  ASSERT_EQ(minOfVectorElementsParallel->validation(), true);
  minOfVectorElementsParallel->pre_processing();
  minOfVectorElementsParallel->run();
  minOfVectorElementsParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(minOfVectorElementsParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(extrema, global_min[0]);
  }
}

TEST(zaitsev_a_min_of_vector_elements_mpi, test_task_run) {
  const int extrema = -1000;
  const int minRangeValue = -500;
  const int maxRangeValue = 500;

  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_min(1, maxRangeValue + 1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 10e6;
    global_vec = zaitsev_a_min_of_vector_elements_mpi::getRandomVector(count_size_vector, minRangeValue, maxRangeValue);
    global_vec[count_size_vector / 2] = extrema;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  auto minOfVectorElementsParallel =
      std::make_shared<zaitsev_a_min_of_vector_elements_mpi::MinOfVectorElementsParallel>(taskDataPar);
  ASSERT_EQ(minOfVectorElementsParallel->validation(), true);
  minOfVectorElementsParallel->pre_processing();
  minOfVectorElementsParallel->run();
  minOfVectorElementsParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(minOfVectorElementsParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(extrema, global_min[0]);
  }
}
