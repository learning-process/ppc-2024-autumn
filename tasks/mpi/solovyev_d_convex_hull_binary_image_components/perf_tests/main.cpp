#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/solovyev_d_convex_hull_binary_image_components/include/header.hpp"

namespace solovyev_d_convex_hull_binary_image_components_mpi {
std::vector<int> generateVector(int sz) {
  std::vector<int> vec(sz);
  for (int j = 0; j < 10; j++) {
    for (int i = j * sz / 10; i < (j + 1) * sz / 10; i++) {
      vec[i] = j % 2;
    }
  }
  return vec;
}
}  // namespace solovyev_d_convex_hull_binary_image_components_mpi
TEST(solovyev_d_convex_hull_binary_image_components_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int dimX = 500;
  int dimY = 500;
  std::vector<int> in = solovyev_d_convex_hull_binary_image_components_mpi::generateVector(500 * 500);
  std::vector<std::vector<int>> expected = {{499, 99, 499, 50, 0, 50, 0, 99},
                                            {499, 199, 499, 150, 0, 150, 0, 199},
                                            {499, 299, 499, 250, 0, 250, 0, 299},
                                            {499, 399, 499, 350, 0, 350, 0, 399},
                                            {499, 499, 499, 450, 0, 450, 0, 499}};
  std::vector<std::vector<int>> out = {std::vector<int>(expected[0].size(), 0), std::vector<int>(expected[1].size(), 0),
                                       std::vector<int>(expected[2].size(), 0), std::vector<int>(expected[3].size(), 0),
                                       std::vector<int>(expected[4].size(), 0)};
  // Create data
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimX));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimY));
    taskDataPar->inputs_count.emplace_back(in.size());
    for (size_t i = 0; i < out.size(); i++) {
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
      taskDataPar->outputs_count.emplace_back(out.size());
    }
  }
  // Create Task
  auto testTaskMPI =
      std::make_shared<solovyev_d_convex_hull_binary_image_components_mpi::ConvexHullBinaryImageComponentsMPI>(
          taskDataPar);
  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskMPI);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(expected, out);
  }
}

TEST(solovyev_d_convex_hull_binary_image_components_mpi, test_task_run) {
  boost::mpi::communicator world;
  int dimX = 500;
  int dimY = 500;
  std::vector<int> in = solovyev_d_convex_hull_binary_image_components_mpi::generateVector(500 * 500);
  std::vector<std::vector<int>> expected = {{499, 99, 499, 50, 0, 50, 0, 99},
                                            {499, 199, 499, 150, 0, 150, 0, 199},
                                            {499, 299, 499, 250, 0, 250, 0, 299},
                                            {499, 399, 499, 350, 0, 350, 0, 399},
                                            {499, 499, 499, 450, 0, 450, 0, 499}};
  std::vector<std::vector<int>> out = {std::vector<int>(expected[0].size(), 0), std::vector<int>(expected[1].size(), 0),
                                       std::vector<int>(expected[2].size(), 0), std::vector<int>(expected[3].size(), 0),
                                       std::vector<int>(expected[4].size(), 0)};
  // Create data
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimX));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimY));
    taskDataPar->inputs_count.emplace_back(in.size());
    for (size_t i = 0; i < out.size(); i++) {
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
      taskDataPar->outputs_count.emplace_back(out.size());
    }
  }
  // Create Task
  auto testTaskMPI =
      std::make_shared<solovyev_d_convex_hull_binary_image_components_mpi::ConvexHullBinaryImageComponentsMPI>(
          taskDataPar);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskMPI);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(expected, out);
  }
}