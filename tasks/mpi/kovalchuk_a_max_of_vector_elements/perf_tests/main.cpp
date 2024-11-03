// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <limits>
#include <random>
#include <vector>
#include <algorithm>

#include "core/perf/include/perf.hpp"
#include "mpi/kovalchuk_a_max_of_vector_elements/include/ops_mpi.hpp"

using namespace kovalchuk_a_max_of_vector_elements_mpi;
TEST(kovalchuk_a_max_of_vector_elements_mpi, test_task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    std::vector<std::vector<int>> matrix;
    std::vector<int32_t> max_value(1, std::numeric_limits<int>::min());
    int reference = std::numeric_limits<int>::max();
    std::random_device dev;
    std::mt19937 gen(dev());

    int rows = 9;
    int columns = 9;
    int start_gen = -99;
    int fin_gen = 99;
    matrix = getRandomMatrix(rows, columns, start_gen, fin_gen);
    std::uniform_int_distribution<int> row_dist(0, rows - 1);
    std::uniform_int_distribution<int> col_dist(0, columns - 1);
    int random_row = row_dist(gen);
    int random_col = col_dist(gen);
    matrix[random_row][random_col] = reference;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(max_value.data()));
    taskDataSeq->outputs_count.emplace_back(max_value.size());

    auto testTaskMPI = std::make_shared<TestTaskMPI>(taskDataSeq);

    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perfAttr->current_timer = [&] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
      return static_cast<double>(duration) * 1e-9;
    };

    auto perfResults = std::make_shared<ppc::core::PerfResults>();
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskMPI);
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
    ppc::core::Perf::print_perf_statistic(perfResults);

    ASSERT_EQ(reference, max_value[0]);
  }
}

TEST(kovalchuk_a_max_of_vector_elements_mpi, test_pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    std::vector<std::vector<int>> matrix;
    std::vector<int32_t> max_value(1, std::numeric_limits<int>::min());
    int reference = std::numeric_limits<int>::max();
    std::random_device dev;
    std::mt19937 gen(dev());

    int rows = 9999;
    int columns = 9999;
    int start_gen = -999;
    int fin_gen = 999;
    matrix = getRandomMatrix(rows, columns, start_gen, fin_gen);
    std::uniform_int_distribution<int> row_dist(0, rows - 1);
    std::uniform_int_distribution<int> col_dist(0, columns - 1);
    int random_row = row_dist(gen);
    int random_col = col_dist(gen);
    matrix[random_row][random_col] = reference;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(max_value.data()));
    taskDataSeq->outputs_count.emplace_back(max_value.size());

    auto testTaskMPI = std::make_shared<TestTaskMPI>(taskDataSeq);

    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perfAttr->current_timer = [&] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
      return static_cast<double>(duration) * 1e-9;
    };

    auto perfResults = std::make_shared<ppc::core::PerfResults>();
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskMPI);
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
    ppc::core::Perf::print_perf_statistic(perfResults);

    ASSERT_EQ(reference, max_value[0]);
  }
}

