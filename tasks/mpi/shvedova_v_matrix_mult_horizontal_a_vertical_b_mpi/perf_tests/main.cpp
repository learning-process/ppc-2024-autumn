#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi/include/ops_mpi.hpp"

namespace shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi {

std::vector<int> getRandomMatrix(int rows, int cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dist(-1000, 1000);
  std::vector<int> matrix(rows * cols);
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = dist(gen);
  }
  return matrix;
}

}  // namespace shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi, pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> global_matrix_a;
  std::vector<int> global_matrix_b;
  std::vector<int> global_result_seq;
  std::vector<int> global_result_par;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int rowA = 100;
  int colA = 100;
  int colB = 100;

  if (world.rank() == 0) {
    global_matrix_a = shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::getRandomMatrix(rowA, colA);
    global_matrix_b = shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::getRandomMatrix(colA, colB);

    global_result_seq.resize(rowA * colB, 0);
    global_result_par.resize(rowA * colB, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix_a.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix_a.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix_b.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix_b.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rowA));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&colA));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&colB));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result_par.data()));
    taskDataPar->outputs_count.emplace_back(global_result_seq.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix_a.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix_a.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix_b.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix_b.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rowA));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&colA));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&colB));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result_seq.data()));
    taskDataSeq->outputs_count.emplace_back(global_result_seq.size());
  }

  auto taskParallel =
      std::make_shared<shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::MatrixMultiplicationTaskParallel>(
          taskDataPar);
  ASSERT_EQ(taskParallel->validation(), true);
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    auto taskSequential =
        std::make_shared<shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::MatrixMultiplicationTaskSequential>(
            taskDataSeq);
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

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(global_result_seq, global_result_par);
  }
}

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi, task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> global_matrix_a;
  std::vector<int> global_matrix_b;
  std::vector<int> global_result_seq;
  std::vector<int> global_result_par;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int rowA = 100;
  int colA = 100;
  int colB = 100;

  if (world.rank() == 0) {
    global_matrix_a = shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::getRandomMatrix(rowA, colA);
    global_matrix_b = shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::getRandomMatrix(colA, colB);

    global_result_seq.resize(rowA * colB, 0);
    global_result_par.resize(rowA * colB, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix_a.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix_a.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix_b.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix_b.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rowA));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&colA));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&colB));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result_par.data()));
    taskDataPar->outputs_count.emplace_back(global_result_seq.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix_a.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix_a.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix_b.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix_b.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rowA));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&colA));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&colB));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result_seq.data()));
    taskDataSeq->outputs_count.emplace_back(global_result_seq.size());
  }

  auto taskParallel =
      std::make_shared<shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::MatrixMultiplicationTaskParallel>(
          taskDataPar);
  ASSERT_EQ(taskParallel->validation(), true);
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    auto taskSequential =
        std::make_shared<shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::MatrixMultiplicationTaskSequential>(
            taskDataSeq);
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

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(global_result_seq, global_result_par);
  }
}
