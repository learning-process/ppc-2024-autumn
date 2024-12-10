#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/ermolaev_v_allreduce_my/include/ops_mpi.hpp"

namespace ermolaev_v_allreduce_mpi {

template <typename _T, typename _S>
std::vector<_T> getRandomMatrix(_S rows, _S cols, _T min, _T max) {
  std::random_device dev;
  std::mt19937 gen(dev());

  std::vector<_T> matrix(rows * cols);

  const auto gen_max = (double)std::numeric_limits<uint32_t>::max();
  const _T range = max - min + 1;

  for (_S i = 0; i < rows; i++) {
    for (_S j = 0; j < cols; j++) {
      matrix[i * cols + j] = min + (_T)(gen() / gen_max * range);
    }
  }

  return matrix;
}

template <typename value_type>
void fillData(std::shared_ptr<ppc::core::TaskData>& taskData, std::vector<value_type>& matrix,
              std::vector<value_type>& res, uint32_t rows, uint32_t cols) {
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskData->inputs_count.emplace_back(rows);
  taskData->inputs_count.emplace_back(cols);

  res.resize(matrix.size());

  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  taskData->outputs_count.emplace_back(rows);
  taskData->outputs_count.emplace_back(cols);
};

template <typename parallel_task_class, typename value_type>
void funcTestBody(uint32_t rows, uint32_t cols, value_type gen_min, value_type gen_max) {
  boost::mpi::communicator world;
  std::vector<value_type> matrix;
  std::vector<value_type> mpi_res;

  if (world.rank() == 0) std::cout << "Run test with " << rows << "x" << cols << " matrix\n";

  auto run = [](ppc::core::Task& task) {
    task.validation();
    task.pre_processing();
    task.run();
    task.post_processing();
  };

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = ermolaev_v_allreduce_mpi::getRandomMatrix<value_type, int32_t>(rows, cols, gen_min, gen_max);
    fillData(taskDataPar, matrix, mpi_res, rows, cols);
  }

  parallel_task_class testMpiTaskParallel(taskDataPar);
  run(testMpiTaskParallel);

  if (world.rank() == 0) {
    std::vector<value_type> seq_res;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    fillData(taskDataSeq, matrix, seq_res, rows, cols);

    typename ermolaev_v_allreduce_mpi::TestMPITaskSequential<value_type, uint32_t> testMpiTaskSequential(taskDataSeq);
    run(testMpiTaskSequential);

    for (uint32_t i = 0; i < rows; i++)
      for (uint32_t j = 0; j < cols; j++) ASSERT_NEAR(seq_res[i * cols + j], mpi_res[i * cols + j], 1e-1);

    std::cout << "Successful test with " << rows << "x" << cols << " matrix\n";
  }
}

template <typename parallel_task_class, typename value_type>
void perfTestBody(uint32_t rows, uint32_t cols, ppc::core::PerfResults::TypeOfRunning type) {
  boost::mpi::communicator world;
  std::vector<value_type> matrix;
  std::vector<value_type> res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(rows * cols, 1);
    fillData(taskDataPar, matrix, res, rows, cols);
  }

  auto task = std::make_shared<parallel_task_class>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);

  if (type == ppc::core::PerfResults::PIPELINE)
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
  else if (type == ppc::core::PerfResults::TASK_RUN)
    perfAnalyzer->task_run(perfAttr, perfResults);
  else if (type == ppc::core::PerfResults::NONE)
    perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    if (type != ppc::core::PerfResults::NONE) ppc::core::Perf::print_perf_statistic(perfResults);
    for (uint32_t i = 0; i < rows; i++)
      for (uint32_t j = 0; j < cols; j++) ASSERT_EQ(res[i * cols + j], 0);
  }
}

template <typename parallel_task_class, typename value_type>
void testValidation() {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    auto matrix = ermolaev_v_allreduce_mpi::getRandomMatrix(3, 3, -3, 3);
    std::vector<value_type> res(matrix.size());

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    parallel_task_class task(taskDataPar);
    ASSERT_FALSE(task.validation());

    parallel_task_class task1(taskDataPar);
    taskDataPar->inputs_count.emplace_back(3);
    taskDataPar->inputs_count.emplace_back(3);
    ASSERT_FALSE(task1.validation());

    parallel_task_class task2(taskDataPar);
    taskDataPar->outputs_count.emplace_back(3);
    taskDataPar->outputs_count.emplace_back(3);
    ASSERT_FALSE(task2.validation());

    parallel_task_class task3(taskDataPar);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    ASSERT_FALSE(task3.validation());

    parallel_task_class task4(taskDataPar);
    taskDataPar->inputs.clear();
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    ASSERT_FALSE(task4.validation());

    parallel_task_class task5(taskDataPar);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    ASSERT_TRUE(task5.validation());
  }
}

}  // namespace ermolaev_v_allreduce_mpi