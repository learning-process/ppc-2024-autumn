// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/rezantseva_a_simple_iteration_method/include/ops_mpi_rezantseva.hpp"

static int offset = 0;

std::pair<std::vector<double>, std::vector<double>> rezantseva_a_simple_iteration_method_mpi::createRandomMatrix(
    size_t n) {
  std::vector<double> A(n * n);
  std::vector<double> b(n);
  std::mt19937 gen;
  gen.seed((unsigned)time(nullptr) + ++offset);

  for (size_t i = 0; i < n; i++) {
    double sum = 0.0;

    // gen non diagonal elements
    for (size_t j = 0; j < n; j++) {
      if (i != j) {
        A[i * n + j] = static_cast<double>(gen() % 50 - 25);  // from -25 to 24
        sum += std::abs(A[i * n + j]);
      }
    }
    A[i * n + i] = sum + static_cast<double>(gen() % 50 + 1);
    b[i] = static_cast<double>(gen() % 100);  // from 0 to 99
  }
  return {A, b};
}

TEST(rezantseva_a_simple_iteration_method_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  size_t size = 8000;
  auto [A, b] = rezantseva_a_simple_iteration_method_mpi::createRandomMatrix(size);
  std::vector<double> out(size, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(b.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  // Create Task
  auto simpleIterationMPI = std::make_shared<rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI>(taskDataPar);

  ASSERT_EQ(simpleIterationMPI->validation(), true);
  simpleIterationMPI->pre_processing();
  simpleIterationMPI->run();
  simpleIterationMPI->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  //  Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(simpleIterationMPI);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(size, out.size());
  }
}

TEST(rezantseva_a_simple_iteration_method_mpi, test_task_run) {
  boost::mpi::communicator world;
  size_t size = 8000;
  auto [A, b] = rezantseva_a_simple_iteration_method_mpi::createRandomMatrix(size);
  std::vector<double> out(size, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(b.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  // Create Task
  auto simpleIterationMPI = std::make_shared<rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI>(taskDataPar);

  ASSERT_EQ(simpleIterationMPI->validation(), true);
  simpleIterationMPI->pre_processing();
  simpleIterationMPI->run();
  simpleIterationMPI->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  //  Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(simpleIterationMPI);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(size, out.size());
  }
}
