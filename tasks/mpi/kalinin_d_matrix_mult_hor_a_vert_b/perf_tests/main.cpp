#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kalinin_d_matrix_mult_hor_a_vert_b/include/ops_mpi.hpp"

namespace kalinin_d_matrix_mult_hor_a_vert_b_mpi {

std::vector<int> getRandomMatrix(int rows, int columns) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dist(-1000, 1000);
  std::vector<int> matrix(rows * columns);

  for (int i = 0; i < rows * columns; i++) {
    matrix[i] = dist(gen);
  }

  return matrix;
}

}  // namespace kalinin_d_matrix_mult_hor_a_vert_b_mpi

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator _world;

  std::vector<int> global_matrix_a;
  std::vector<int> global_matrix_b;
  std::vector<int> global_result_seq;
  std::vector<int> global_result_par;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int rowsA = 100;
  int columnsA = 100;
  int columnsB = 100;

  if (_world.rank() == 0) {
    global_matrix_a = kalinin_d_matrix_mult_hor_a_vert_b_mpi::getRandomMatrix(rowsA, columnsA);
    global_matrix_b = kalinin_d_matrix_mult_hor_a_vert_b_mpi::getRandomMatrix(columnsA, columnsB);

    global_result_seq.resize(rowsA * columnsB, 0);
    global_result_par.resize(rowsA * columnsB, 0);

    auto add_task_data = [&global_matrix_a, &global_matrix_b, &rowsA, &columnsA, &columnsB](
                             std::shared_ptr<ppc::core::TaskData>& taskData, std::vector<int>& result) {
      std::vector<std::pair<void*, size_t>> inputs = {{global_matrix_a.data(), global_matrix_a.size()},
                                                      {global_matrix_b.data(), global_matrix_b.size()},
                                                      {&rowsA, 1},
                                                      {&columnsA, 1},
                                                      {&columnsB, 1}};

      for (const auto& [ptr, size] : inputs) {
        taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(ptr));
        taskData->inputs_count.emplace_back(size);
      }

      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
      taskData->outputs_count.emplace_back(result.size());
    };

    add_task_data(taskDataPar, global_result_par);
    add_task_data(taskDataSeq, global_result_seq);
  }

  auto taskParallel =
      std::make_shared<kalinin_d_matrix_mult_hor_a_vert_b_mpi::ParallelMatrixMultiplicationTask>(taskDataPar);
  ASSERT_EQ(taskParallel->validation(), true);
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (_world.rank() == 0) {
    auto taskSequential =
        std::make_shared<kalinin_d_matrix_mult_hor_a_vert_b_mpi::SequentialMatrixMultiplicationTask>(taskDataSeq);
    ASSERT_EQ(taskSequential->validation(), true);
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();
  }

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (_world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(global_result_seq, global_result_par);
  }
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator _world;

  std::vector<int> global_matrix_a;
  std::vector<int> global_matrix_b;
  std::vector<int> global_result_seq;
  std::vector<int> global_result_par;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int rowsA = 100;
  int columnsA = 100;
  int columnsB = 100;

  if (_world.rank() == 0) {
    global_matrix_a = kalinin_d_matrix_mult_hor_a_vert_b_mpi::getRandomMatrix(rowsA, columnsA);
    global_matrix_b = kalinin_d_matrix_mult_hor_a_vert_b_mpi::getRandomMatrix(columnsA, columnsB);

    global_result_seq.resize(rowsA * columnsB, 0);
    global_result_par.resize(rowsA * columnsB, 0);

    auto add_task_data = [&global_matrix_a, &global_matrix_b, &rowsA, &columnsA, &columnsB](
                             std::shared_ptr<ppc::core::TaskData>& taskData, std::vector<int>& result) {
      std::vector<std::pair<void*, size_t>> inputs = {{global_matrix_a.data(), global_matrix_a.size()},
                                                      {global_matrix_b.data(), global_matrix_b.size()},
                                                      {&rowsA, 1},
                                                      {&columnsA, 1},
                                                      {&columnsB, 1}};

      for (const auto& [ptr, size] : inputs) {
        taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(ptr));
        taskData->inputs_count.emplace_back(size);
      }

      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
      taskData->outputs_count.emplace_back(result.size());
    };

    add_task_data(taskDataPar, global_result_par);
    add_task_data(taskDataSeq, global_result_seq);
  }

  auto taskParallel =
      std::make_shared<kalinin_d_matrix_mult_hor_a_vert_b_mpi::ParallelMatrixMultiplicationTask>(taskDataPar);
  ASSERT_EQ(taskParallel->validation(), true);
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (_world.rank() == 0) {
    auto taskSequential =
        std::make_shared<kalinin_d_matrix_mult_hor_a_vert_b_mpi::SequentialMatrixMultiplicationTask>(taskDataSeq);
    ASSERT_EQ(taskSequential->validation(), true);
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();
  }

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (_world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(global_result_seq, global_result_par);
  }
}
