#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <memory>
#include <vector>

#include "mpi/shulpin_i_strip_hA_vB/include/strip_hA_vB.hpp"
/*
TEST(shulpin_strip_scheme_A_B, matrix_1x1) {
  boost::mpi::communicator world;

  int cols_a = 1;
  int rows_a = 1;
  int cols_b = 1;
  int rows_b = 1;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res_mpi;
  std::vector<int> global_res_seq;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A = shulpin_strip_scheme_A_B::get_RND_matrix(cols_a, rows_a);
    global_B = shulpin_strip_scheme_A_B::get_RND_matrix(cols_b, rows_b);
    global_res_mpi.resize(cols_b * rows_a, 0);
    global_res_seq.resize(cols_b * rows_a, 0);

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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(global_res_seq.size());
  }

  auto taskParallel = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_par>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();
  if (world.rank() == 0) {
    auto taskSequential = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_seq>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();
    ASSERT_EQ(global_res_mpi, global_res_seq);
  }
}

TEST(shulpin_strip_scheme_A_B, matrix_2x2) {
  boost::mpi::communicator world;

  int cols_a = 2;
  int rows_a = 2;
  int cols_b = 2;
  int rows_b = 2;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res_mpi;
  std::vector<int> global_res_seq;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A = shulpin_strip_scheme_A_B::get_RND_matrix(cols_a, rows_a);
    global_B = shulpin_strip_scheme_A_B::get_RND_matrix(cols_b, rows_b);
    global_res_mpi.resize(cols_b * rows_a, 0);
    global_res_seq.resize(cols_b * rows_a, 0);

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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(global_res_seq.size());
  }

  auto taskParallel = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_par>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();
  if (world.rank() == 0) {
    auto taskSequential = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_seq>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();
    ASSERT_EQ(global_res_mpi, global_res_seq);
  }
}

TEST(shulpin_strip_scheme_A_B, matrix_3x3) {
  boost::mpi::communicator world;

  int cols_a = 3;
  int rows_a = 3;
  int cols_b = 3;
  int rows_b = 3;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res_mpi;
  std::vector<int> global_res_seq;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A = shulpin_strip_scheme_A_B::get_RND_matrix(cols_a, rows_a);
    global_B = shulpin_strip_scheme_A_B::get_RND_matrix(cols_b, rows_b);
    global_res_mpi.resize(cols_b * rows_a, 0);
    global_res_seq.resize(cols_b * rows_a, 0);

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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(global_res_seq.size());
  }

  auto taskParallel = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_par>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    auto taskSequential = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_seq>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();
    ASSERT_EQ(global_res_mpi, global_res_seq);
  }
}

TEST(shulpin_strip_scheme_A_B, matrix_4x4) {
  boost::mpi::communicator world;

  int cols_a = 4;
  int rows_a = 4;
  int cols_b = 4;
  int rows_b = 4;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res_mpi;
  std::vector<int> global_res_seq;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A = shulpin_strip_scheme_A_B::get_RND_matrix(cols_a, rows_a);
    global_B = shulpin_strip_scheme_A_B::get_RND_matrix(cols_b, rows_b);
    global_res_mpi.resize(cols_b * rows_a, 0);
    global_res_seq.resize(cols_b * rows_a, 0);

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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(global_res_seq.size());
  }

  auto taskParallel = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_par>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    auto taskSequential = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_seq>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();
    ASSERT_EQ(global_res_mpi, global_res_seq);
  }
}
*/
TEST(shulpin_strip_scheme_A_B, matrix_5x5) {
  boost::mpi::communicator world;

  int cols_a = 5;
  int rows_a = 5;
  int cols_b = 5;
  int rows_b = 5;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res_mpi;
  std::vector<int> global_res_seq;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A = shulpin_strip_scheme_A_B::get_RND_matrix(cols_a, rows_a);
    global_B = shulpin_strip_scheme_A_B::get_RND_matrix(cols_b, rows_b);
    global_res_mpi.resize(cols_b * rows_a, 0);
    global_res_seq.resize(cols_b * rows_a, 0);

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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(global_res_seq.size());
  }

  auto taskParallel = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_par> (taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    auto taskSequential = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_seq> (taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();
    ASSERT_EQ(global_res_mpi, global_res_seq);
  }
}

TEST(shulpin_strip_scheme_A_B, matrix_100x100) {
  boost::mpi::communicator world;

  int cols_a = 100;
  int rows_a = 100;
  int cols_b = 100;
  int rows_b = 100;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res_mpi;
  std::vector<int> global_res_seq;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A = shulpin_strip_scheme_A_B::get_RND_matrix(cols_a, rows_a);
    global_B = shulpin_strip_scheme_A_B::get_RND_matrix(cols_b, rows_b);
    global_res_mpi.resize(cols_b * rows_a, 0);
    global_res_seq.resize(cols_b * rows_a, 0);

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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(global_res_seq.size());
  }

  auto taskParallel = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_par>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    auto taskSequential = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_seq>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();
    ASSERT_EQ(global_res_mpi, global_res_seq);
  }
}

TEST(shulpin_strip_scheme_A_B, matrix_500x500) {
  boost::mpi::communicator world;

  int cols_a = 500;
  int rows_a = 500;
  int cols_b = 500;
  int rows_b = 500;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res_mpi;
  std::vector<int> global_res_seq;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A = shulpin_strip_scheme_A_B::get_RND_matrix(cols_a, rows_a);
    global_B = shulpin_strip_scheme_A_B::get_RND_matrix(cols_b, rows_b);
    global_res_mpi.resize(cols_b * rows_a, 0);
    global_res_seq.resize(cols_b * rows_a, 0);

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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(global_res_seq.size());
  }

  auto taskParallel = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_par>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    auto taskSequential = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_seq>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();
    ASSERT_EQ(global_res_mpi, global_res_seq);
  }
}

TEST(shulpin_strip_scheme_A_B, matrix_5x4_n_4x5) {
  boost::mpi::communicator world;

  int cols_a = 4;
  int rows_a = 5;
  int cols_b = 5;
  int rows_b = 4;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res_mpi;
  std::vector<int> global_res_seq;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A = shulpin_strip_scheme_A_B::get_RND_matrix(cols_a, rows_a);
    global_B = shulpin_strip_scheme_A_B::get_RND_matrix(cols_b, rows_b);
    global_res_mpi.resize(cols_b * rows_a, 0);
    global_res_seq.resize(cols_b * rows_a, 0);

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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(global_res_seq.size());
  }

  auto taskParallel = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_par>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    auto taskSequential = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_seq>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();
    ASSERT_EQ(global_res_mpi, global_res_seq);
  }
}

TEST(shulpin_strip_scheme_A_B, matrix_4x5_5x4) {
  boost::mpi::communicator world;

  int cols_a = 5;
  int rows_a = 4;
  int cols_b = 4;
  int rows_b = 5;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res_mpi;
  std::vector<int> global_res_seq;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A = shulpin_strip_scheme_A_B::get_RND_matrix(cols_a, rows_a);
    global_B = shulpin_strip_scheme_A_B::get_RND_matrix(cols_b, rows_b);
    global_res_mpi.resize(cols_b * rows_a, 0);
    global_res_seq.resize(cols_b * rows_a, 0);

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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(global_res_seq.size());
  }

  auto taskParallel = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_par>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    auto taskSequential = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_seq>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();
    ASSERT_EQ(global_res_mpi, global_res_seq);
  }
}

TEST(shulpin_strip_scheme_A_B, matrix_4x2_2x5) {
  boost::mpi::communicator world;

  int cols_a = 2;
  int rows_a = 4;
  int cols_b = 5;
  int rows_b = 2;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res_mpi;
  std::vector<int> global_res_seq;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A = shulpin_strip_scheme_A_B::get_RND_matrix(cols_a, rows_a);
    global_B = shulpin_strip_scheme_A_B::get_RND_matrix(cols_b, rows_b);
    global_res_mpi.resize(cols_b * rows_a, 0);
    global_res_seq.resize(cols_b * rows_a, 0);

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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(global_res_seq.size());
  }

  auto taskParallel = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_par>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    auto taskSequential = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_seq>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();
    ASSERT_EQ(global_res_mpi, global_res_seq);
  }
}

TEST(shulpin_strip_scheme_A_B, matrix_1x10_10x1) {
  boost::mpi::communicator world;

  int cols_a = 10;
  int rows_a = 1;
  int cols_b = 1;
  int rows_b = 10;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res_mpi;
  std::vector<int> global_res_seq;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A = shulpin_strip_scheme_A_B::get_RND_matrix(cols_a, rows_a);
    global_B = shulpin_strip_scheme_A_B::get_RND_matrix(cols_b, rows_b);
    global_res_mpi.resize(cols_b * rows_a, 0);
    global_res_seq.resize(cols_b * rows_a, 0);

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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(global_res_seq.size());
  }

  auto taskParallel = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_par>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    auto taskSequential = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_seq>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();
    ASSERT_EQ(global_res_mpi, global_res_seq);
  }
}

TEST(shulpin_strip_scheme_A_B, matrix_10x1_1x10) {
  boost::mpi::communicator world;

  int cols_a = 1;
  int rows_a = 10;
  int cols_b = 10;
  int rows_b = 1;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res_mpi;
  std::vector<int> global_res_seq;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A = shulpin_strip_scheme_A_B::get_RND_matrix(cols_a, rows_a);
    global_B = shulpin_strip_scheme_A_B::get_RND_matrix(cols_b, rows_b);
    global_res_mpi.resize(cols_b * rows_a, 0);
    global_res_seq.resize(cols_b * rows_a, 0);

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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(global_res_seq.size());
  }

  auto taskParallel = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_par>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    auto taskSequential = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_seq>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();
    ASSERT_EQ(global_res_mpi, global_res_seq);
  }
}

TEST(shulpin_strip_scheme_A_B, matrix_10x5_5x20) {
  boost::mpi::communicator world;

  int cols_a = 5;
  int rows_a = 10;
  int cols_b = 20;
  int rows_b = 5;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res_mpi;
  std::vector<int> global_res_seq;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A = shulpin_strip_scheme_A_B::get_RND_matrix(cols_a, rows_a);
    global_B = shulpin_strip_scheme_A_B::get_RND_matrix(cols_b, rows_b);
    global_res_mpi.resize(cols_b * rows_a, 0);
    global_res_seq.resize(cols_b * rows_a, 0);

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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(global_res_seq.size());
  }

  auto taskParallel = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_par>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    auto taskSequential = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_seq>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();
    ASSERT_EQ(global_res_mpi, global_res_seq);
  }
}

TEST(shulpin_strip_scheme_A_B, invalid_matrix) {
  boost::mpi::communicator world;

  int cols_a = 1;
  int rows_a = 2;
  int cols_b = 3;
  int rows_b = 4;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res_mpi;
  std::vector<int> global_res_seq;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A = shulpin_strip_scheme_A_B::get_RND_matrix(cols_a, rows_a);
    global_B = shulpin_strip_scheme_A_B::get_RND_matrix(cols_b, rows_b);
    global_res_mpi.resize(cols_b * rows_a, 0);
    global_res_seq.resize(cols_b * rows_a, 0);

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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows_b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(global_res_seq.size());
  }

  auto taskParallel = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_par>(taskDataPar);
  ASSERT_FALSE(taskParallel->validation());

  if (world.rank() == 0) {
    auto taskSequential = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_seq>(taskDataSeq);
    ASSERT_FALSE(taskSequential->validation());
  }
}