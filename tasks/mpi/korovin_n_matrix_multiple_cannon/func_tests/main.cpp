#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/korovin_n_matrix_multiple_cannon/include/ops_mpi.hpp"

TEST(korovin_n_matrix_multiple_cannon_mpi, matrix_1x1) {
  boost::mpi::communicator world;
  int numRowsA = 1;
  int numColsA_RowsB = 1;
  int numColsB = 1;
  std::vector<double> A = {7};
  std::vector<double> B = {3};
  std::vector<double> expected_C = {21};
  std::vector<double> C_mpi(numRowsA * numColsB, 0.0);
  std::vector<double> C_seq(numRowsA * numColsB, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(numRowsA);
    taskData->inputs_count.emplace_back(numColsA_RowsB);
    taskData->inputs_count.emplace_back(numColsB);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_mpi.data()));
  }

  korovin_n_matrix_multiple_cannon_mpi::TestMPITaskParallel testTask(taskData);
  ASSERT_TRUE(testTask.validation());
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs_count.emplace_back(numRowsA);
    taskDataSeq->inputs_count.emplace_back(numColsA_RowsB);
    taskDataSeq->inputs_count.emplace_back(numColsB);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_seq.data()));

    korovin_n_matrix_multiple_cannon_mpi::TestMPITaskSequential testTaskSeq(taskDataSeq);
    ASSERT_TRUE(testTaskSeq.validation());
    testTaskSeq.pre_processing();
    testTaskSeq.run();
    testTaskSeq.post_processing();

    for (int i = 0; i < (int)C_mpi.size(); i++) {
      EXPECT_DOUBLE_EQ(C_mpi[i], expected_C[i]);
      EXPECT_DOUBLE_EQ(C_mpi[i], C_seq[i]);
    }
  }
}

TEST(korovin_n_matrix_multiple_cannon_mpi, matrix_2x2) {
  boost::mpi::communicator world;
  int numRowsA = 2;
  int numColsA_RowsB = 2;
  int numColsB = 2;
  std::vector<double> A = {1, 2, 3, 4};
  std::vector<double> B = {5, 6, 7, 8};
  std::vector<double> expected_C = {19, 22, 43, 50};
  std::vector<double> C_mpi(numRowsA * numColsB, 0.0);
  std::vector<double> C_seq(numRowsA * numColsB, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(numRowsA);
    taskData->inputs_count.emplace_back(numColsA_RowsB);
    taskData->inputs_count.emplace_back(numColsB);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_mpi.data()));
  }

  korovin_n_matrix_multiple_cannon_mpi::TestMPITaskParallel testTask(taskData);
  ASSERT_TRUE(testTask.validation());
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs_count.emplace_back(numRowsA);
    taskDataSeq->inputs_count.emplace_back(numColsA_RowsB);
    taskDataSeq->inputs_count.emplace_back(numColsB);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_seq.data()));

    korovin_n_matrix_multiple_cannon_mpi::TestMPITaskSequential testTaskSeq(taskDataSeq);
    ASSERT_TRUE(testTaskSeq.validation());
    testTaskSeq.pre_processing();
    testTaskSeq.run();
    testTaskSeq.post_processing();

    for (int i = 0; i < (int)C_mpi.size(); i++) {
      EXPECT_DOUBLE_EQ(C_mpi[i], expected_C[i]);
      EXPECT_DOUBLE_EQ(C_mpi[i], C_seq[i]);
    }
  }
}

TEST(korovin_n_matrix_multiple_cannon_mpi, matrix_3x3) {
  boost::mpi::communicator world;
  int numRowsA = 3;
  int numColsA_RowsB = 3;
  int numColsB = 3;
  std::vector<double> A = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<double> B = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<double> expected_C = {15, 18, 21, 42, 54, 66, 69, 90, 111};
  std::vector<double> C_mpi(numRowsA * numColsB, 0.0);
  std::vector<double> C_seq(numRowsA * numColsB, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(numRowsA);
    taskData->inputs_count.emplace_back(numColsA_RowsB);
    taskData->inputs_count.emplace_back(numColsB);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_mpi.data()));
  }

  korovin_n_matrix_multiple_cannon_mpi::TestMPITaskParallel testTask(taskData);
  ASSERT_TRUE(testTask.validation());
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs_count.emplace_back(numRowsA);
    taskDataSeq->inputs_count.emplace_back(numColsA_RowsB);
    taskDataSeq->inputs_count.emplace_back(numColsB);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_seq.data()));

    korovin_n_matrix_multiple_cannon_mpi::TestMPITaskSequential testTaskSeq(taskDataSeq);
    ASSERT_TRUE(testTaskSeq.validation());
    testTaskSeq.pre_processing();
    testTaskSeq.run();
    testTaskSeq.post_processing();

    for (int i = 0; i < (int)C_mpi.size(); i++) {
      EXPECT_DOUBLE_EQ(C_mpi[i], expected_C[i]);
      EXPECT_DOUBLE_EQ(C_mpi[i], C_seq[i]);
    }
  }
}

TEST(korovin_n_matrix_multiple_cannon_mpi, matrix_2x3_3x2) {
  boost::mpi::communicator world;
  int numRowsA = 2;
  int numColsA_RowsB = 3;
  int numColsB = 2;
  std::vector<double> A = {1, 2, 3, 4, 5, 6};
  std::vector<double> B = {7, 8, 9, 10, 11, 12};
  std::vector<double> expected_C = {58, 64, 139, 154};
  std::vector<double> C_mpi(numRowsA * numColsB, 0.0);
  std::vector<double> C_seq(numRowsA * numColsB, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(numRowsA);
    taskData->inputs_count.emplace_back(numColsA_RowsB);
    taskData->inputs_count.emplace_back(numColsB);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_mpi.data()));
  }

  korovin_n_matrix_multiple_cannon_mpi::TestMPITaskParallel testTask(taskData);
  ASSERT_TRUE(testTask.validation());
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs_count.emplace_back(numRowsA);
    taskDataSeq->inputs_count.emplace_back(numColsA_RowsB);
    taskDataSeq->inputs_count.emplace_back(numColsB);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_seq.data()));

    korovin_n_matrix_multiple_cannon_mpi::TestMPITaskSequential testTaskSeq(taskDataSeq);
    ASSERT_TRUE(testTaskSeq.validation());
    testTaskSeq.pre_processing();
    testTaskSeq.run();
    testTaskSeq.post_processing();

    for (int i = 0; i < (int)C_mpi.size(); i++) {
      EXPECT_DOUBLE_EQ(C_mpi[i], expected_C[i]);
      EXPECT_DOUBLE_EQ(C_mpi[i], C_seq[i]);
    }
  }
}

TEST(korovin_n_matrix_multiple_cannon_mpi, matrix_6x4_4x5) {
  boost::mpi::communicator world;
  int numRowsA = 6;
  int numColsA_RowsB = 4;
  int numColsB = 5;
  std::vector<double> A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  std::vector<double> B = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
  std::vector<double> expected_C = {110, 120, 130, 140, 150, 246, 272, 298, 324, 350, 382, 424, 466, 508,  550,
                                    518, 576, 634, 692, 750, 654, 728, 802, 876, 950, 790, 880, 970, 1060, 1150};
  std::vector<double> C_mpi(numRowsA * numColsB, 0.0);
  std::vector<double> C_seq(numRowsA * numColsB, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(numRowsA);
    taskData->inputs_count.emplace_back(numColsA_RowsB);
    taskData->inputs_count.emplace_back(numColsB);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_mpi.data()));
  }

  korovin_n_matrix_multiple_cannon_mpi::TestMPITaskParallel testTask(taskData);
  ASSERT_TRUE(testTask.validation());
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs_count.emplace_back(numRowsA);
    taskDataSeq->inputs_count.emplace_back(numColsA_RowsB);
    taskDataSeq->inputs_count.emplace_back(numColsB);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_seq.data()));

    korovin_n_matrix_multiple_cannon_mpi::TestMPITaskSequential testTaskSeq(taskDataSeq);
    ASSERT_TRUE(testTaskSeq.validation());
    testTaskSeq.pre_processing();
    testTaskSeq.run();
    testTaskSeq.post_processing();

    for (int i = 0; i < (int)C_mpi.size(); i++) {
      EXPECT_DOUBLE_EQ(C_mpi[i], expected_C[i]);
      EXPECT_DOUBLE_EQ(C_mpi[i], C_seq[i]);
    }
  }
}

TEST(korovin_n_matrix_multiple_cannon_mpi, matrix_4x4) {
  boost::mpi::communicator world;
  int numRowsA = 4;
  int numColsA_RowsB = 4;
  int numColsB = 4;
  std::vector<double> A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> B = {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
  std::vector<double> expected_C = {250, 260,  270,  280,  618,  644,  670,  696,
                                    986, 1028, 1070, 1112, 1354, 1412, 1470, 1528};
  std::vector<double> C_mpi(numRowsA * numColsB, 0.0);
  std::vector<double> C_seq(numRowsA * numColsB, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(numRowsA);
    taskData->inputs_count.emplace_back(numColsA_RowsB);
    taskData->inputs_count.emplace_back(numColsB);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_mpi.data()));
  }

  korovin_n_matrix_multiple_cannon_mpi::TestMPITaskParallel testTask(taskData);
  ASSERT_TRUE(testTask.validation());
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs_count.emplace_back(numRowsA);
    taskDataSeq->inputs_count.emplace_back(numColsA_RowsB);
    taskDataSeq->inputs_count.emplace_back(numColsB);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_seq.data()));

    korovin_n_matrix_multiple_cannon_mpi::TestMPITaskSequential testTaskSeq(taskDataSeq);
    ASSERT_TRUE(testTaskSeq.validation());
    testTaskSeq.pre_processing();
    testTaskSeq.run();
    testTaskSeq.post_processing();

    for (int i = 0; i < (int)C_mpi.size(); i++) {
      EXPECT_DOUBLE_EQ(C_mpi[i], expected_C[i]);
      EXPECT_DOUBLE_EQ(C_mpi[i], C_seq[i]);
    }
  }
}

TEST(korovin_n_matrix_multiple_cannon_mpi, matrix_2x3_3x2_negative) {
  boost::mpi::communicator world;
  int numRowsA = 2;
  int numColsA_RowsB = 3;
  int numColsB = 2;
  std::vector<double> A = {1, -2, 0, -3, 4, 5};
  std::vector<double> B = {0, 6, -7, 8, 9, -10};
  std::vector<double> expected_C = {14, -10, 17, -36};
  std::vector<double> C_mpi(numRowsA * numColsB, 0.0);
  std::vector<double> C_seq(numRowsA * numColsB, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(numRowsA);
    taskData->inputs_count.emplace_back(numColsA_RowsB);
    taskData->inputs_count.emplace_back(numColsB);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_mpi.data()));
  }

  korovin_n_matrix_multiple_cannon_mpi::TestMPITaskParallel testTask(taskData);
  ASSERT_TRUE(testTask.validation());
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs_count.emplace_back(numRowsA);
    taskDataSeq->inputs_count.emplace_back(numColsA_RowsB);
    taskDataSeq->inputs_count.emplace_back(numColsB);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_seq.data()));

    korovin_n_matrix_multiple_cannon_mpi::TestMPITaskSequential testTaskSeq(taskDataSeq);
    ASSERT_TRUE(testTaskSeq.validation());
    testTaskSeq.pre_processing();
    testTaskSeq.run();
    testTaskSeq.post_processing();

    for (int i = 0; i < (int)C_mpi.size(); i++) {
      EXPECT_DOUBLE_EQ(C_mpi[i], expected_C[i]);
      EXPECT_DOUBLE_EQ(C_mpi[i], C_seq[i]);
    }
  }
}

TEST(korovin_n_matrix_multiple_cannon_mpi, matrix_zeros) {
  boost::mpi::communicator world;
  int numRowsA = 2;
  int numColsA_RowsB = 2;
  int numColsB = 2;
  std::vector<double> A = {0, 0, 0, 0};
  std::vector<double> B = {0, 0, 0, 0};
  std::vector<double> expected_C = {0, 0, 0, 0};
  std::vector<double> C_mpi(numRowsA * numColsB, 0.0);
  std::vector<double> C_seq(numRowsA * numColsB, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(numRowsA);
    taskData->inputs_count.emplace_back(numColsA_RowsB);
    taskData->inputs_count.emplace_back(numColsB);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_mpi.data()));
  }

  korovin_n_matrix_multiple_cannon_mpi::TestMPITaskParallel testTask(taskData);
  ASSERT_TRUE(testTask.validation());
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs_count.emplace_back(numRowsA);
    taskDataSeq->inputs_count.emplace_back(numColsA_RowsB);
    taskDataSeq->inputs_count.emplace_back(numColsB);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_seq.data()));

    korovin_n_matrix_multiple_cannon_mpi::TestMPITaskSequential testTaskSeq(taskDataSeq);
    ASSERT_TRUE(testTaskSeq.validation());
    testTaskSeq.pre_processing();
    testTaskSeq.run();
    testTaskSeq.post_processing();

    for (int i = 0; i < (int)C_mpi.size(); i++) {
      EXPECT_DOUBLE_EQ(C_mpi[i], expected_C[i]);
      EXPECT_DOUBLE_EQ(C_mpi[i], C_seq[i]);
    }
  }
}

TEST(korovin_n_matrix_multiple_cannon_mpi, matrix_validation_empty) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count = {0, 0, 0};
  korovin_n_matrix_multiple_cannon_mpi::TestMPITaskParallel testTask(taskData);
  if (world.rank() == 0) {
    ASSERT_FALSE(testTask.validation());
  }
}

TEST(korovin_n_matrix_multiple_cannon_mpi, matrix_validation_zero_dimensions) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count = {0, 3, 2};
  taskData->inputs = {nullptr, nullptr};
  korovin_n_matrix_multiple_cannon_mpi::TestMPITaskParallel testTask(taskData);
  if (world.rank() == 0) {
    ASSERT_FALSE(testTask.validation());
  }
}

TEST(korovin_n_matrix_multiple_cannon_mpi, matrix_validation_miss_pointers) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count = {2, 2, 2};
  taskData->inputs = {nullptr, nullptr};
  korovin_n_matrix_multiple_cannon_mpi::TestMPITaskParallel testTask(taskData);
  if (world.rank() == 0) {
    ASSERT_FALSE(testTask.validation());
  }
}

TEST(korovin_n_matrix_multiple_cannon_mpi, matrix_validation_inputs) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count = {2, 2};
  std::vector<double> A = {1, 2, 3, 4};
  taskData->inputs = {reinterpret_cast<uint8_t*>(A.data())};
  korovin_n_matrix_multiple_cannon_mpi::TestMPITaskParallel testTask(taskData);
  if (world.rank() == 0) {
    ASSERT_FALSE(testTask.validation());
  }
}

TEST(korovin_n_matrix_multiple_cannon_mpi, matrix_3x3_inversion) {
  boost::mpi::communicator world;
  int numRowsA = 3;
  int numColsA_RowsB = 3;
  int numColsB = 3;

  std::vector<double> A = {1, 2, 3, 0, 1, 4, 0, 0, 1};
  std::vector<double> A_inv = {1, -2, 5, 0, 1, -4, 0, 0, 1};
  std::vector<double> expected_C = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<double> C_mpi(numRowsA * numColsB, 0.0);
  std::vector<double> C_seq(numRowsA * numColsB, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(numRowsA);
    taskData->inputs_count.emplace_back(numColsA_RowsB);
    taskData->inputs_count.emplace_back(numColsB);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_inv.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_mpi.data()));
  }

  korovin_n_matrix_multiple_cannon_mpi::TestMPITaskParallel testTask(taskData);
  ASSERT_TRUE(testTask.validation());
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs_count.emplace_back(numRowsA);
    taskDataSeq->inputs_count.emplace_back(numColsA_RowsB);
    taskDataSeq->inputs_count.emplace_back(numColsB);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_inv.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_seq.data()));

    korovin_n_matrix_multiple_cannon_mpi::TestMPITaskSequential testTaskSeq(taskDataSeq);
    ASSERT_TRUE(testTaskSeq.validation());
    testTaskSeq.pre_processing();
    testTaskSeq.run();
    testTaskSeq.post_processing();

    for (int i = 0; i < (int)C_mpi.size(); i++) {
      EXPECT_DOUBLE_EQ(C_mpi[i], expected_C[i]);
      EXPECT_DOUBLE_EQ(C_mpi[i], C_seq[i]);
    }
  }
}
