#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <random>
#include <vector>

#include "mpi/budazhapova_e_matrix_multiplication/include/matrix_mult_mpi.hpp"

namespace budazhapova_e_matrix_mult_mpi {
std::vector<int> generate_random_vector_or_matrix(int size, int minValue, int maxValue) {
  std::vector<int> randomVector(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(minValue, maxValue);
  for (int i = 0; i < size; ++i) {
    randomVector[i] = dis(gen);
  }

  return randomVector;
}
}  // namespace budazhapova_e_matrix_mult_mpi

TEST(budazhapova_e_matrix_mult_mpi, ordinary_test_1) {
  boost::mpi::communicator world;
  std::vector<int> A_matrix(12, 1);
  std::vector<int> b_vector(4, 1);
  std::vector<int> out(3, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_matrix.data()));
    taskDataPar->inputs_count.emplace_back(A_matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(b_vector.data()));
    taskDataPar->inputs_count.emplace_back(b_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  budazhapova_e_matrix_mult_mpi::MatrixMultParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> out_seq(3, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(A_matrix.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b_vector.data()));
    taskDataSeq->inputs_count.emplace_back(b_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    budazhapova_e_matrix_mult_mpi::MatrixMultSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(out, out_seq);
  }
}

TEST(budazhapova_e_matrix_mult_mpi, ordinary_test_2_for_three_proc_to_crash) {
  boost::mpi::communicator world;
  std::vector<int> A_matrix(20, 3);
  std::vector<int> b_vector(5, 1);
  std::vector<int> out(4, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_matrix.data()));
    taskDataPar->inputs_count.emplace_back(A_matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(b_vector.data()));
    taskDataPar->inputs_count.emplace_back(b_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  budazhapova_e_matrix_mult_mpi::MatrixMultParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> out_seq(4, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(A_matrix.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b_vector.data()));
    taskDataSeq->inputs_count.emplace_back(b_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    budazhapova_e_matrix_mult_mpi::MatrixMultSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(out, out_seq);
  }
}

TEST(budazhapova_e_matrix_mult_mpi, ordinary_test_3_for_two_or_four_proc_to_crash) {
  boost::mpi::communicator world;
  std::vector<int> A_matrix(45, 2);
  std::vector<int> b_vector(9, 1);
  std::vector<int> out(5, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_matrix.data()));
    taskDataPar->inputs_count.emplace_back(A_matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(b_vector.data()));
    taskDataPar->inputs_count.emplace_back(b_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  budazhapova_e_matrix_mult_mpi::MatrixMultParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> out_seq(5, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(A_matrix.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b_vector.data()));
    taskDataSeq->inputs_count.emplace_back(b_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    budazhapova_e_matrix_mult_mpi::MatrixMultSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(out, out_seq);
  }
}
TEST(budazhapova_e_matrix_mult_mpi, random_test_1) {
  boost::mpi::communicator world;
  int size_m;
  int size_v;
  int minn;
  int maxx;
  size_m = 100;
  size_v = 5;
  minn = 1;
  maxx = 12;
  std::vector<int> A_matrix = budazhapova_e_matrix_mult_mpi::generate_random_vector_or_matrix(size_m, minn, maxx);
  std::vector<int> b_vector = budazhapova_e_matrix_mult_mpi::generate_random_vector_or_matrix(size_v, minn, maxx);
  std::vector<int> out(20, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_matrix.data()));
    taskDataPar->inputs_count.emplace_back(A_matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(b_vector.data()));
    taskDataPar->inputs_count.emplace_back(b_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  budazhapova_e_matrix_mult_mpi::MatrixMultParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> out_seq(20, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(A_matrix.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b_vector.data()));
    taskDataSeq->inputs_count.emplace_back(b_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    budazhapova_e_matrix_mult_mpi::MatrixMultSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(out, out_seq);
  }
}

TEST(budazhapova_e_matrix_mult_mpi, validation_test_1) {
  boost::mpi::communicator world;
  std::vector<int> A_matrix(12, 1);
  std::vector<int> b_vector(5, 1);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_matrix.data()));
    taskDataPar->inputs_count.emplace_back(A_matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(b_vector.data()));
    taskDataPar->inputs_count.emplace_back(b_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
    budazhapova_e_matrix_mult_mpi::MatrixMultParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(budazhapova_e_matrix_mult_mpi, validation_test_2) {
  boost::mpi::communicator world;
  std::vector<int> A_matrix(12, 1);
  std::vector<int> b_vector = {};
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_matrix.data()));
    taskDataPar->inputs_count.emplace_back(A_matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(b_vector.data()));
    taskDataPar->inputs_count.emplace_back(b_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
    budazhapova_e_matrix_mult_mpi::MatrixMultParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}
