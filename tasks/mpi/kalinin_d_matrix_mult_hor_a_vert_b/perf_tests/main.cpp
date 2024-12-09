#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kalinin_d_matrix_mult_hor_a_vert_b/include/ops_mpi.hpp"

namespace kalinin_d_matrix_mult_hor_a_vert_b_mpi {

std::vector<int> get_random_matrix(int rows, int cols, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dist(min, max);
  std::vector<int> matrix(rows * cols);
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = dist(gen);
  }
  return matrix;
}

}  // namespace kalinin_d_matrix_mult_hor_a_vert_b_mpi

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res_seq;
  std::vector<int> global_res_par;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int rows_A = 200;
  int columns_A = 200;
  int rows_B = 200;
  int columns_B = 200;

  if (world.rank() == 0) {
    global_A = kalinin_d_matrix_mult_hor_a_vert_b_mpi::get_random_matrix(rows_A, columns_A, -1000, 1000);
    global_B = kalinin_d_matrix_mult_hor_a_vert_b_mpi::get_random_matrix(rows_B, columns_B, -1000, 1000);

    global_res_seq.resize(rows_A * columns_B, 0);
    global_res_par.resize(rows_A * columns_B, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(rows_A);
    taskDataPar->inputs_count.emplace_back(columns_A);
    taskDataPar->inputs_count.emplace_back(columns_A);
    taskDataPar->inputs_count.emplace_back(columns_B);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res_par.data()));
    taskDataPar->outputs_count.emplace_back(global_res_par.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_B.data()));
    taskDataSeq->inputs_count.emplace_back(rows_A);
    taskDataSeq->inputs_count.emplace_back(columns_A);
    taskDataSeq->inputs_count.emplace_back(columns_A);
    taskDataSeq->inputs_count.emplace_back(columns_B);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(global_res_seq.size());
  }

  auto taskParallel = std::make_shared<kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(taskParallel->validation(), true);
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    auto taskSequential = std::make_shared<kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskSequential>(taskDataSeq);
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
    EXPECT_EQ(global_res_seq, global_res_par);
  }
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res_seq;
  std::vector<int> global_res_par;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int rows_A = 200;
  int columns_A = 200;
  int rows_B = 200;
  int columns_B = 200;

  if (world.rank() == 0) {
    global_A = kalinin_d_matrix_mult_hor_a_vert_b_mpi::get_random_matrix(rows_A, columns_A, -1000, 1000);
    global_B = kalinin_d_matrix_mult_hor_a_vert_b_mpi::get_random_matrix(rows_B, columns_B, -1000, 1000);

    global_res_seq.resize(rows_A * columns_B, 0);
    global_res_par.resize(rows_A * columns_B, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(rows_A);
    taskDataPar->inputs_count.emplace_back(columns_A);
    taskDataPar->inputs_count.emplace_back(columns_A);
    taskDataPar->inputs_count.emplace_back(columns_B);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res_par.data()));
    taskDataPar->outputs_count.emplace_back(global_res_par.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_B.data()));
    taskDataSeq->inputs_count.emplace_back(rows_A);
    taskDataSeq->inputs_count.emplace_back(columns_A);
    taskDataSeq->inputs_count.emplace_back(columns_A);
    taskDataSeq->inputs_count.emplace_back(columns_B);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(global_res_seq.size());
  }

  auto taskParallel = std::make_shared<kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    auto taskSequential = std::make_shared<kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskSequential>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
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
    EXPECT_EQ(global_res_seq, global_res_par);
  }
}