#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/volochaev_s_vertical_ribbon_scheme_16/include/ops_mpi.hpp"

namespace volochaev_s_vertical_ribbon_scheme_16_mpi {

void get_random_matrix(std::vector<int> &mat) {
  std::random_device dev;
  std::mt19937 gen(dev());

  for (size_t i = 0; i < mat.size(); ++i) {
    mat[i] = 50 - gen() % 100;
  }
}

}  // namespace volochaev_s_vertical_ribbon_scheme_16_mpi

TEST(volochaev_s_vertical_ribbon_scheme_16_mpi, Test_0) {
  boost::mpi::communicator world;
  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int m = 0;

  if (world.rank() == 0) {
    global_B = {0, 0, 0};

    global_res.resize(m, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(global_A.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(global_B.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto taskParallel = std::make_shared<volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_mpi>(taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel->validation());
  } else {
    EXPECT_TRUE(taskParallel->validation());
  }
}

TEST(volochaev_s_vertical_ribbon_scheme_16_mpi, Test_1) {
  boost::mpi::communicator world;
  std::vector<int> global_A(100, 0);
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int m = 0;

  if (world.rank() == 0) {
    global_B = {0, 0, 0};

    global_res.resize(m, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(global_A.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(global_B.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto taskParallel = std::make_shared<volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_mpi>(taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel->validation());
  } else {
    EXPECT_TRUE(taskParallel->validation());
  }
}

TEST(volochaev_s_vertical_ribbon_scheme_16_mpi, Test_2) {
  boost::mpi::communicator world;
  std::vector<int> global_A(100, 0);
  std::vector<int> global_B(3, 0);
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int m = 0;

  if (world.rank() == 0) {
    global_res.resize(m, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(global_A.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(global_B.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto taskParallel = std::make_shared<volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_mpi>(taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel->validation());
  } else {
    EXPECT_TRUE(taskParallel->validation());
  }
}

TEST(volochaev_s_vertical_ribbon_scheme_16_mpi, Test_3) {
  boost::mpi::communicator world;

  std::vector<int> global_A(10);
  std::vector<int> global_B(2);

  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_A);
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_B);

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

  std::vector<int> global_A(20);
  std::vector<int> global_B(2);
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_A);
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_B);

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

  std::vector<int> global_A(100);
  std::vector<int> global_B(25);
  ;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_A);
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_B);

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

  std::vector<int> global_A(100);
  std::vector<int> global_B(1);
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_A);
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_B);

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

  std::vector<int> global_A(289);
  std::vector<int> global_B(17);
  ;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_A);
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_B);

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

  std::vector<int> global_A(100);
  std::vector<int> global_B(100);
  ;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_A);
    volochaev_s_vertical_ribbon_scheme_16_mpi::get_random_matrix(global_B);

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