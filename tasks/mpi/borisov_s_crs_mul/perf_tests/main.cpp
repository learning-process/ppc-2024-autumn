// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/borisov_s_crs_mul/include/ops_mpi.hpp"

static void generateRandomCRSMatrix(int rows, int cols, double sparsity, std::vector<double>& values,
                                    std::vector<int>& col_index, std::vector<int>& row_ptr) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis_value(-10.0, 10.0);
  std::uniform_real_distribution<> dis_sparse(0.0, 1.0);

  row_ptr.resize(rows + 1, 0);
  for (int i = 0; i < rows; ++i) {
    row_ptr[i + 1] = row_ptr[i];
    for (int j = 0; j < cols; ++j) {
      if (dis_sparse(gen) < sparsity) {
        values.push_back(dis_value(gen));
        col_index.push_back(j);
        ++row_ptr[i + 1];
      }
    }
  }
}

TEST(MPI_CRS_Matrix_Perf_Test, Test_Pipeline_Run) {
  boost::mpi::communicator world;

  std::vector<double> A_values;
  std::vector<double> B_values;
  std::vector<int> A_col_index;
  std::vector<int> B_col_index;
  std::vector<int> A_row_ptr;
  std::vector<int> B_row_ptr;

  if (world.rank() == 0) {
    generateRandomCRSMatrix(1000, 1000, 0.01, A_values, A_col_index, A_row_ptr);
    generateRandomCRSMatrix(1000, 1000, 0.01, B_values, B_col_index, B_row_ptr);
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_col_index.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_row_ptr.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_col_index.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_row_ptr.data()));

    taskDataPar->inputs_count = {
        static_cast<unsigned int>(A_values.size()),    static_cast<unsigned int>(A_col_index.size()),
        static_cast<unsigned int>(A_row_ptr.size()),   static_cast<unsigned int>(B_values.size()),
        static_cast<unsigned int>(B_col_index.size()), static_cast<unsigned int>(B_row_ptr.size())};

    std::vector<double> C_values;
    std::vector<int> C_col_index;
    std::vector<int> C_row_ptr(1001, 0);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_values.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_col_index.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_row_ptr.data()));

    taskDataPar->outputs_count = {static_cast<unsigned int>(C_values.size()),
                                  static_cast<unsigned int>(C_col_index.size()),
                                  static_cast<unsigned int>(C_row_ptr.size())};
  }

  auto mpiTask = std::make_shared<borisov_s_crs_mul_mpi::CrsMatrixMulTaskMPI>(taskDataPar);
  ASSERT_TRUE(mpiTask->validation());
  mpiTask->pre_processing();
  mpiTask->run();
  mpiTask->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(mpiTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(MPI_CRS_Matrix_Perf_Test, Test_Task_Run) {
  boost::mpi::communicator world;

  std::vector<double> A_values;
  std::vector<double> B_values;
  std::vector<int> A_col_index;
  std::vector<int> B_col_index;
  std::vector<int> A_row_ptr;
  std::vector<int> B_row_ptr;

  if (world.rank() == 0) {
    generateRandomCRSMatrix(1000, 1000, 0.01, A_values, A_col_index, A_row_ptr);
    generateRandomCRSMatrix(1000, 1000, 0.01, B_values, B_col_index, B_row_ptr);
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_col_index.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_row_ptr.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_col_index.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_row_ptr.data()));

    taskDataPar->inputs_count = {
        static_cast<unsigned int>(A_values.size()),    static_cast<unsigned int>(A_col_index.size()),
        static_cast<unsigned int>(A_row_ptr.size()),   static_cast<unsigned int>(B_values.size()),
        static_cast<unsigned int>(B_col_index.size()), static_cast<unsigned int>(B_row_ptr.size())};

    std::vector<double> C_values;
    std::vector<int> C_col_index;
    std::vector<int> C_row_ptr(1001, 0);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_values.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_col_index.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_row_ptr.data()));

    taskDataPar->outputs_count = {static_cast<unsigned int>(C_values.size()),
                                  static_cast<unsigned int>(C_col_index.size()),
                                  static_cast<unsigned int>(C_row_ptr.size())};
  }

  auto mpiTask = std::make_shared<borisov_s_crs_mul_mpi::CrsMatrixMulTaskMPI>(taskDataPar);
  ASSERT_TRUE(mpiTask->validation());
  mpiTask->pre_processing();
  mpiTask->run();
  mpiTask->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(mpiTask);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}