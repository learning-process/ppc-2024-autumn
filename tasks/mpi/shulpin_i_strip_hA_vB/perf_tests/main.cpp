#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/shulpin_i_strip_hA_vB/include/strip_hA_vB.hpp"

TEST(shulpin_strip_scheme_A_B, pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res_mpi;
  std::vector<int> global_res_seq;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  
  int cols_a = 500;
  int rows_a = 500;
  int cols_b = 500;
  int rows_b = 500;

  if (world.rank() == 0) {
    global_A = shulpin_strip_scheme_A_B::get_RND_matrix(rows_a, cols_a);
    global_B = shulpin_strip_scheme_A_B::get_RND_matrix(rows_b, cols_b);

    global_res_seq.resize(rows_a * cols_b, 0);
    global_res_mpi.resize(rows_a * cols_b, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(global_A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(global_B.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_a));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows_a));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_b));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows_b));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res_mpi.data()));
    taskDataPar->outputs_count.emplace_back(global_res_mpi.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(global_A.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_B.data()));
    taskDataSeq->inputs_count.emplace_back(global_B.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_a));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows_a));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_a));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(global_res_seq.size());
  }

  auto taskParallel = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_par>(taskDataPar);
  ASSERT_EQ(taskParallel->validation(), true);
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    auto taskSequential = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_seq>(taskDataSeq);
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
    ASSERT_EQ(global_res_mpi, global_res_seq);
  }
}

TEST(shulpin_strip_scheme_A_B, task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res_mpi;
  std::vector<int> global_res_seq;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int cols_a = 500;
  int rows_a = 500;
  int cols_b = 500;
  int rows_b = 500;

  if (world.rank() == 0) {
    global_A = shulpin_strip_scheme_A_B::get_RND_matrix(rows_a, cols_a);
    global_B = shulpin_strip_scheme_A_B::get_RND_matrix(rows_b, cols_b);

    global_res_mpi.resize(rows_a * cols_b, 0);
    global_res_seq.resize(rows_a * cols_b, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(global_A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(global_B.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_a));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows_a));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_b));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows_b));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res_mpi.data()));
    taskDataPar->outputs_count.emplace_back(global_res_mpi.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(global_A.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_B.data()));
    taskDataSeq->inputs_count.emplace_back(global_B.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_a));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(global_res_seq.size());
  }

  auto taskParallel = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_par>(taskDataPar);
  ASSERT_EQ(taskParallel->validation(), true);
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    auto taskSequential = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_seq>(taskDataSeq);
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
    ASSERT_EQ(global_res_mpi, global_res_seq);
  }
}