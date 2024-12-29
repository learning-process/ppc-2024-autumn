#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/volochaev_s_vertical_ribbon_scheme_16/include/ops_mpi.hpp"

namespace volochaev_s_vertical_ribbon_scheme_16_mpi {

void get_random_matrix(std::vector<int> &mat, int a, int b) {
  std::random_device dev;
  std::mt19937 gen(dev());

  if (a >= b) {
    throw std::invalid_argument("error.");
  }

  std::uniform_int_distribution<> dis(a, b);

  for (size_t i = 0; i < mat.size(); ++i) {
    mat[i] = dis(gen);
  }
}

}  // namespace volochaev_s_vertical_ribbon_scheme_16_mpi

TEST(volochaev_s_vertical_ribbon_scheme_16_mpi, Test_mines_1) {
  boost::mpi::communicator world;
  std::vector<int> global_A(100);

  ASSERT_ANY_THROW(volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_A, 90, -100));
}

TEST(volochaev_s_vertical_ribbon_scheme_16_mpi, Test_0) {
  boost::mpi::communicator world;
  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int m = 0;

  if (world.rank() == 0) {
    global_B = {0, 0, 0};

    global_res.resize(m, 0);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(global_A.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataSeq->inputs_count.emplace_back(global_B.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataSeq->outputs_count.emplace_back(global_res.size());

    volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_seq testMpiTaskSequential(taskDataSeq);
    EXPECT_FALSE(testMpiTaskSequential.validation());
  }
}

TEST(volochaev_s_vertical_ribbon_scheme_16_mpi, Test_1) {
  boost::mpi::communicator world;
  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int m = 0;

  if (world.rank() == 0) {
    global_A.resize(100, 0);
    global_res.resize(m, 0);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(global_A.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataSeq->inputs_count.emplace_back(global_B.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataSeq->outputs_count.emplace_back(global_res.size());

    volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_seq testMpiTaskSequential(taskDataSeq);
    EXPECT_FALSE(testMpiTaskSequential.validation());
  }
}

TEST(volochaev_s_vertical_ribbon_scheme_16_mpi, Test_2) {
  boost::mpi::communicator world;
  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int m = 0;

  if (world.rank() == 0) {
    global_res.resize(m, 0);
    global_A.resize(100, 0);
    global_B.resize(3, 0);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(global_A.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataSeq->inputs_count.emplace_back(global_B.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataSeq->outputs_count.emplace_back(global_res.size());

    volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_seq testMpiTaskSequential(taskDataSeq);
    EXPECT_FALSE(testMpiTaskSequential.validation());
  }
}

TEST(volochaev_s_vertical_ribbon_scheme_16_mpi, Test_3) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;

  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(10);
    global_B.resize(2);
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_A, -100, 100);
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_B, -200, 20);

    global_res.resize(5, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(global_A.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(global_B.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_mpi taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res(5, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(global_A.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataSeq->inputs_count.emplace_back(global_B.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_res.data()));
    taskDataSeq->outputs_count.emplace_back(expected_res.size());

    // Create Task
    volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_seq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_res, expected_res);
  }
}

TEST(volochaev_s_vertical_ribbon_scheme_16_mpi, Test_4) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(20);
    global_B.resize(2);
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_A, -100, 200);
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_B, -10, 10);

    global_res.resize(10, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(global_A.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(global_B.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_mpi taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res(10, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(global_A.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataSeq->inputs_count.emplace_back(global_B.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_res.data()));
    taskDataSeq->outputs_count.emplace_back(expected_res.size());

    // Create Task
    volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_seq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_res, expected_res);
  }
}

TEST(volochaev_s_vertical_ribbon_scheme_16_mpi, Test_5) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;
  ;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(100);
    global_B.resize(25);
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_A, -1000, 1000);
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_B, -2, 400);

    global_res.resize(4, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(global_A.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(global_B.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_mpi taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res(4, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(global_A.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataSeq->inputs_count.emplace_back(global_B.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_res.data()));
    taskDataSeq->outputs_count.emplace_back(expected_res.size());

    // Create Task
    volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_seq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_res, expected_res);
  }
}

TEST(volochaev_s_vertical_ribbon_scheme_16_mpi, Test_6) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(100);
    global_B.resize(1);
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_A, -100, 100);
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_B, 12, 48);

    global_res.resize(100, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(global_A.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(global_B.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_mpi taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res(100, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(global_A.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataSeq->inputs_count.emplace_back(global_B.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_res.data()));
    taskDataSeq->outputs_count.emplace_back(expected_res.size());

    // Create Task
    volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_seq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_res, expected_res);
  }
}

TEST(volochaev_s_vertical_ribbon_scheme_16_mpi, Test_7) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;
  ;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(289);
    global_B.resize(17);
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_A, -100, 1000);
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_B, 20, 8000);

    global_res.resize(17, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(global_A.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(global_B.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_mpi taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res(17, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(global_A.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataSeq->inputs_count.emplace_back(global_B.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_res.data()));
    taskDataSeq->outputs_count.emplace_back(expected_res.size());

    // Create Task
    volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_seq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_res, expected_res);
  }
}

TEST(volochaev_s_vertical_ribbon_scheme_16_mpi, Test_8) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;
  ;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(100);
    global_B.resize(100);
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_A, -100, 100);
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_B, 20, 100);

    global_res.resize(1, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(global_A.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(global_B.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_mpi taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(global_A.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataSeq->inputs_count.emplace_back(global_B.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_res.data()));
    taskDataSeq->outputs_count.emplace_back(expected_res.size());

    // Create Task
    volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_seq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_res, expected_res);
  }
}