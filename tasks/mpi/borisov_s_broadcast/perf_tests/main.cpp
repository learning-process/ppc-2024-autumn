#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/borisov_s_broadcast/include/ops_mpi.hpp"

TEST(parallel_distance_matrix_perf_test, test_pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  const size_t num_points = 1000;

  std::vector<double> global_points;
  std::vector<double> global_distance_matrix;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_points = borisov_s_broadcast::getRandomPoints(num_points);
    global_distance_matrix.resize(num_points * num_points, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_points.data()));
    taskDataPar->inputs_count.emplace_back(num_points);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_distance_matrix.data()));
    taskDataPar->outputs_count.emplace_back(global_distance_matrix.size());
  } else {
    global_points.resize(num_points * 2);
    global_distance_matrix.resize(num_points * num_points, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_points.data()));
    taskDataPar->inputs_count.emplace_back(num_points);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_distance_matrix.data()));
    taskDataPar->outputs_count.emplace_back(global_distance_matrix.size());
  }

  auto testTaskParallel = std::make_shared<borisov_s_broadcast::DistanceMatrixTaskParallel>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(global_distance_matrix.size(), static_cast<size_t>(num_points * num_points));
  }
}

TEST(parallel_distance_matrix_perf_test, test_task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  const size_t num_points = 1000;

  std::vector<double> global_points;
  std::vector<double> global_distance_matrix;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_points = borisov_s_broadcast::getRandomPoints(num_points);
    global_distance_matrix.resize(num_points * num_points, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_points.data()));
    taskDataPar->inputs_count.emplace_back(num_points);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_distance_matrix.data()));
    taskDataPar->outputs_count.emplace_back(global_distance_matrix.size());
  } else {
    global_points.resize(num_points * 2);
    global_distance_matrix.resize(num_points * num_points, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_points.data()));
    taskDataPar->inputs_count.emplace_back(num_points);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_distance_matrix.data()));
    taskDataPar->outputs_count.emplace_back(global_distance_matrix.size());
  }

  auto testTaskParallel = std::make_shared<borisov_s_broadcast::DistanceMatrixTaskParallel>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(global_distance_matrix.size(), static_cast<size_t>(num_points * num_points));
  }
}
