#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/ermolaev_v_allreduce/include/shared_ptr_array.hpp"

namespace ermolaev_v_allreduce_mpi {
template <typename _T>
using Matrix = std::vector<ermolaev_v_allreduce_mpi::shared_ptr_array<_T>>;

template <typename _T, typename _S>
Matrix<_T> getRandomMatrix(_S rows, _S cols, _T min, _T max) {
  std::random_device dev;
  std::mt19937 gen(dev());

  Matrix<_T> matrix(rows);
  for (_S i = 0; i < rows; i++) matrix[i] = ermolaev_v_allreduce_mpi::shared_ptr_array<_T>(cols);

  const auto gen_max = (double)std::numeric_limits<uint32_t>::max();
  const _T range = max - min + 1;

  for (_S i = 0; i < rows; i++) {
    for (_S j = 0; j < cols; j++) {
      matrix[i][j] = min + (_T)(gen() / gen_max * range);
    }
  }

  return matrix;
}

template <typename value_type>
void fillData(std::shared_ptr<ppc::core::TaskData>& taskData, Matrix<value_type>& matrix, Matrix<value_type>& res,
              uint32_t rows, uint32_t cols) {
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskData->inputs_count.emplace_back(rows);
  taskData->inputs_count.emplace_back(cols);

  res.resize(rows);
  for (uint32_t i = 0; i < rows; i++) res[i] = ermolaev_v_allreduce_mpi::shared_ptr_array<value_type>(cols);

  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  taskData->outputs_count.emplace_back(rows);
  taskData->outputs_count.emplace_back(cols);
};

template <typename parallel_task_class, typename value_type>
void funcTestBody(uint32_t rows, uint32_t cols, value_type gen_min, value_type gen_max) {
  boost::mpi::communicator world;
  Matrix<value_type> matrix;
  Matrix<value_type> mpi_res;

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
    Matrix<value_type> seq_res;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    fillData(taskDataSeq, matrix, seq_res, rows, cols);

    ermolaev_v_allreduce_mpi::TestMPITaskSequential<value_type, uint32_t> testMpiTaskSequential(taskDataSeq);
    run(testMpiTaskSequential);

    for (uint32_t i = 0; i < rows; i++)
      for (uint32_t j = 0; j < cols; j++) ASSERT_NEAR(seq_res[i][j], mpi_res[i][j], 1e-1);

    std::cout << "Successful test with " << rows << "x" << cols << " matrix\n";
  }
}

template <typename parallel_task_class, typename value_type>
void perfTestBody(uint32_t rows, uint32_t cols, ppc::core::PerfResults::TypeOfRunning type) {
  boost::mpi::communicator world;
  Matrix<value_type> matrix;
  Matrix<value_type> res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(rows);
    for (uint32_t i = 0; i < rows; i++) {
      matrix[i] = ermolaev_v_allreduce_mpi::shared_ptr_array<value_type>(cols);
      std::fill_n(matrix[i].get(), cols, (value_type)1);
    }

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

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (uint32_t i = 0; i < rows; i++)
      for (uint32_t j = 0; j < cols; j++) ASSERT_EQ(res[i][j], 0);
  }
}
}  // namespace ermolaev_v_allreduce_mpi