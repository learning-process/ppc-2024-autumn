#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>

#include "core/perf/include/perf.hpp"
#include "mpi/sedova_o_vertical_ribbon_scheme/include/ops_mpi.hpp"

TEST(sedova_o_vertical_ribbon_scheme_mpi, test_pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_vector;
  std::vector<int> global_result;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int rows_;
  int cols_;
  if (world.rank() == 0) {
    rows_ = 2024;
    cols_ = 2024;
    global_vector.resize(cols_);
    global_matrix.resize(rows_ * cols_);
    for (int j = 0; j < rows_; ++j) {
      for (int i = 0; i < cols_; ++i) {
        global_matrix[j * cols_ + i] = (rand() % 101) - 50;
      }
    }
    for (int i = 0; i < rows_; ++i) {
      global_vector[i] = (rand() % 100) - 50;
    }
    global_result.resize(cols_, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }
  auto taskParallel = std::make_shared<sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    std::vector<int> seq_result(global_result.size(), 0);
    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataSeq->inputs_count.emplace_back(global_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    taskDataSeq->outputs_count.emplace_back(seq_result.size());
    auto taskSequential = std::make_shared<sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();
    ASSERT_EQ(global_result.size(), seq_result.size());
    EXPECT_EQ(global_result, seq_result);
  }
}
TEST(sedova_o_vertical_ribbon_scheme_mpi, test_task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_vector;
  std::vector<int> global_result;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int rows_;
  int cols_;
  if (world.rank() == 0) {
    rows_ = 2000;
    cols_ = 2000;
    global_matrix.resize(rows_ * cols_);
    global_vector.resize(cols_);
    for (int j = 0; j < rows_; ++j) {
      for (int i = 0; i < cols_; ++i) {
        global_matrix[j * cols_ + i] = (rand() % 101) - 50;
      }
    }
    for (int i = 0; i < rows_; ++i) {
      global_vector[i] = (rand() % 100) - 50;
    }
    global_result.resize(cols_, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }
  auto taskParallel = std::make_shared<sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    std::vector<int> seq_result(global_result.size(), 0);
    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataSeq->inputs_count.emplace_back(global_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    taskDataSeq->outputs_count.emplace_back(seq_result.size());
    auto taskSequential = std::make_shared<sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();
    ASSERT_EQ(global_result.size(), seq_result.size());
    EXPECT_EQ(global_result, seq_result);
  }
}