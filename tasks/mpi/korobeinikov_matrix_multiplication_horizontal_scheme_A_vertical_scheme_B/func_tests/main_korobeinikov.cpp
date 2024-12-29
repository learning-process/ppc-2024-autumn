// Copyright 2024 Korobeinikov Arseny
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>

#include "mpi/korobeinikov_matrix_multiplication_horizontal_scheme_A_vertical_scheme_B/include/ops_mpi_korobeinikov.hpp"

namespace korobeinikov_a_test_task_mpi_lab_02 {

std::vector<int> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

}  // namespace korobeinikov_a_test_task_mpi_lab_02

TEST(korobeinikov_matrix_multiplication_horizontal_scheme_A_vertical_scheme_B,
     Test_1_determinate_matrix_3_x_1_and_1_x_3_to_prove_correctness_seq_and_mpi) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> A = {3, 2, 1};
  std::vector<int> B = {1, 2, 3};
  int count_rows_A = 3;
  int count_cols_A = 1;
  int count_rows_B = 1;
  int count_cols_B = 3;

  std::vector<int> out_mpi(9, 0);
  std::vector<int> right_answer = {3, 6, 9, 2, 4, 6, 1, 2, 3};
  int count_rows_out_mpi = 0;
  int count_cols_out_mpi = 0;
  int count_rows_RA = 3;
  int count_cols_RA = 3;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_A));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_A));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(B.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_B));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_B));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_mpi.data()));
    taskDataPar->outputs_count.emplace_back(out_mpi.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_out_mpi));
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_out_mpi));
    taskDataPar->outputs_count.emplace_back(1);
  }

  korobeinikov_a_test_task_mpi_lab_02::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> out_seq(9, 0);
    int count_rows_out_seq = 0;
    int count_cols_out_seq = 0;

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs_count.emplace_back(A.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_A));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_A));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs_count.emplace_back(B.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_B));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_B));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_out_seq));
    taskDataSeq->outputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_out_seq));
    taskDataSeq->outputs_count.emplace_back(1);

    // Create Task
    korobeinikov_a_test_task_mpi_lab_02::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < right_answer.size(); i++) {
      ASSERT_EQ(right_answer[i], out_seq[i]);
    }
    for (size_t i = 0; i < right_answer.size(); i++) {
      ASSERT_EQ(right_answer[i], out_mpi[i]);
    }
    ASSERT_EQ(count_rows_out_seq, count_rows_RA);
    ASSERT_EQ(count_cols_out_seq, count_cols_RA);
    ASSERT_EQ(count_rows_out_mpi, count_rows_RA);
    ASSERT_EQ(count_cols_out_mpi, count_cols_RA);
  }
}

TEST(korobeinikov_matrix_multiplication_horizontal_scheme_A_vertical_scheme_B,
     Test_2_determinate_matrix_3_x_3_and_3_x_2_to_prove_correctness_seq_and_mpi) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> B = {1, 2, 3, 4, 5, 6};
  int count_rows_A = 3;
  int count_cols_A = 3;
  int count_rows_B = 3;
  int count_cols_B = 2;

  std::vector<int> out_mpi(6, 0);
  std::vector<int> right_answer = {22, 28, 49, 64, 76, 100};
  int count_rows_out_mpi = 0;
  int count_cols_out_mpi = 0;
  int count_rows_RA = 3;
  int count_cols_RA = 2;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_A));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_A));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(B.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_B));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_B));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_mpi.data()));
    taskDataPar->outputs_count.emplace_back(out_mpi.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_out_mpi));
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_out_mpi));
    taskDataPar->outputs_count.emplace_back(1);
  }

  korobeinikov_a_test_task_mpi_lab_02::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> out_seq(6, 0);
    int count_rows_out_seq = 0;
    int count_cols_out_seq = 0;

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs_count.emplace_back(A.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_A));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_A));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs_count.emplace_back(B.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_B));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_B));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_out_seq));
    taskDataSeq->outputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_out_seq));
    taskDataSeq->outputs_count.emplace_back(1);

    // Create Task
    korobeinikov_a_test_task_mpi_lab_02::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < right_answer.size(); i++) {
      ASSERT_EQ(right_answer[i], out_seq[i]);
    }
    for (size_t i = 0; i < right_answer.size(); i++) {
      ASSERT_EQ(right_answer[i], out_mpi[i]);
    }
    ASSERT_EQ(count_rows_out_seq, count_rows_RA);
    ASSERT_EQ(count_cols_out_seq, count_cols_RA);
    ASSERT_EQ(count_rows_out_mpi, count_rows_RA);
    ASSERT_EQ(count_cols_out_mpi, count_cols_RA);
  }
}

TEST(korobeinikov_matrix_multiplication_horizontal_scheme_A_vertical_scheme_B,
     Test_3_determinate_matrix_4_x_4_and_4_x_4_to_prove_correctness_seq_and_mpi) {
  boost::mpi::communicator world;

  // Create data
  std::vector<int> A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<int> B = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  int count_rows_A = 4;
  int count_cols_A = 4;
  int count_rows_B = 4;
  int count_cols_B = 4;

  std::vector<int> out_mpi(16, 0);
  int count_rows_out_mpi = 0;
  int count_cols_out_mpi = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_A));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_A));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(B.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_B));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_B));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_mpi.data()));
    taskDataPar->outputs_count.emplace_back(out_mpi.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_out_mpi));
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_out_mpi));
    taskDataPar->outputs_count.emplace_back(1);
  }

  korobeinikov_a_test_task_mpi_lab_02::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> out_seq(16, 0);
    int count_rows_out_seq = 0;
    int count_cols_out_seq = 0;

    std::vector<int> right_answer = {90, 100, 110, 120, 202, 228, 254, 280, 314, 356, 398, 440, 426, 484, 542, 600};
    int count_rows_RA = 4;
    int count_cols_RA = 4;
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs_count.emplace_back(A.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_A));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_A));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs_count.emplace_back(B.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_B));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_B));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_out_seq));
    taskDataSeq->outputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_out_seq));
    taskDataSeq->outputs_count.emplace_back(1);

    // Create Task
    korobeinikov_a_test_task_mpi_lab_02::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < right_answer.size(); i++) {
      ASSERT_EQ(right_answer[i], out_seq[i]);
    }
    for (size_t i = 0; i < right_answer.size(); i++) {
      ASSERT_EQ(right_answer[i], out_mpi[i]);
    }
    ASSERT_EQ(count_rows_out_seq, count_rows_RA);
    ASSERT_EQ(count_cols_out_seq, count_cols_RA);
    ASSERT_EQ(count_rows_out_mpi, count_rows_RA);
    ASSERT_EQ(count_cols_out_mpi, count_cols_RA);
  }
}

TEST(korobeinikov_matrix_multiplication_horizontal_scheme_A_vertical_scheme_B, Test_4_random_matrixes) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> A(9);
  std::vector<int> B(9);
  int count_rows_A = 3;
  int count_cols_A = 3;
  int count_rows_B = 3;
  int count_cols_B = 3;

  std::vector<int> out_mpi(9, 0);
  int count_rows_out_mpi = 0;
  int count_cols_out_mpi = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData
    A = korobeinikov_a_test_task_mpi_lab_02::getRandomVector(9);
    B = korobeinikov_a_test_task_mpi_lab_02::getRandomVector(9);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_A));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_A));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(B.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_B));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_B));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_mpi.data()));
    taskDataPar->outputs_count.emplace_back(out_mpi.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_out_mpi));
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_out_mpi));
    taskDataPar->outputs_count.emplace_back(1);
  }

  korobeinikov_a_test_task_mpi_lab_02::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> out_seq(9, 0);
    int count_rows_out_seq = 0;
    int count_cols_out_seq = 0;

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs_count.emplace_back(A.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_A));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_A));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs_count.emplace_back(B.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_B));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_B));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_out_seq));
    taskDataSeq->outputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_out_seq));
    taskDataSeq->outputs_count.emplace_back(1);

    // Create Task
    korobeinikov_a_test_task_mpi_lab_02::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < out_seq.size(); i++) {
      ASSERT_EQ(out_mpi[i], out_seq[i]);
    }
    ASSERT_EQ(count_rows_out_seq, count_rows_out_mpi);
    ASSERT_EQ(count_cols_out_seq, count_cols_out_mpi);
  }
}

TEST(korobeinikov_matrix_multiplication_horizontal_scheme_A_vertical_scheme_B, Test_5_empty_matrixes) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> A;
  std::vector<int> B;
  int count_rows_A = 0;
  int count_cols_A = 0;
  int count_rows_B = 0;
  int count_cols_B = 0;

  std::vector<int> out_mpi(0, 0);
  int count_rows_out_mpi = 0;
  int count_cols_out_mpi = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_A));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_A));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(B.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_B));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_B));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_mpi.data()));
    taskDataPar->outputs_count.emplace_back(out_mpi.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_out_mpi));
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_out_mpi));
    taskDataPar->outputs_count.emplace_back(1);
  }

  korobeinikov_a_test_task_mpi_lab_02::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> out_seq(0, 0);
    int count_rows_out_seq = 0;
    int count_cols_out_seq = 0;

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs_count.emplace_back(A.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_A));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_A));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs_count.emplace_back(B.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_B));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_B));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_out_seq));
    taskDataSeq->outputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_out_seq));
    taskDataSeq->outputs_count.emplace_back(1);

    // Create Task
    korobeinikov_a_test_task_mpi_lab_02::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < out_seq.size(); i++) {
      ASSERT_EQ(out_mpi[i], out_seq[i]);
    }
    ASSERT_EQ(count_rows_out_seq, count_rows_out_mpi);
    ASSERT_EQ(count_cols_out_seq, count_cols_out_mpi);
  }
}

TEST(korobeinikov_matrix_multiplication_horizontal_scheme_A_vertical_scheme_B, Test_6_validation_false_1) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> A(10);
  std::vector<int> B(10);
  int count_rows_A = 5;
  int count_cols_A = 5;
  int count_rows_B = 5;
  int count_cols_B = 5;

  std::vector<int> out_mpi(0, 0);
  int count_rows_out_mpi = 0;
  int count_cols_out_mpi = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_A));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_A));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(B.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_B));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_B));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_mpi.data()));
    taskDataPar->outputs_count.emplace_back(out_mpi.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_out_mpi));
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_out_mpi));
    taskDataPar->outputs_count.emplace_back(1);
  }

  korobeinikov_a_test_task_mpi_lab_02::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(korobeinikov_matrix_multiplication_horizontal_scheme_A_vertical_scheme_B, Test_7_validation_false_2) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> A(10);
  std::vector<int> B(15);
  int count_rows_A = 5;
  int count_cols_A = 2;
  int count_rows_B = 3;
  int count_cols_B = 5;

  std::vector<int> out_mpi(0, 0);
  int count_rows_out_mpi = 0;
  int count_cols_out_mpi = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_A));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_A));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(B.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_B));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_B));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_mpi.data()));
    taskDataPar->outputs_count.emplace_back(out_mpi.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_out_mpi));
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_out_mpi));
    taskDataPar->outputs_count.emplace_back(1);
  }

  korobeinikov_a_test_task_mpi_lab_02::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}