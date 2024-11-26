#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/smirnov_i_tape_splitting_A/include/ops_mpi.hpp"
namespace smirnov_i_tape_splitting_A {

void get_random_matrix(double* matr, int size) {
  if (size <= 0) {
    throw std::logic_error("wrong matrix size");
  }
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(1, 100);

  for (int i = 0; i < size; i++) {
    matr[i] = distrib(gen);
  }
}

}  // namespace smirnov_i_tape_splitting_A

TEST(smirnov_i_tape_splitting_A_mpi, matrix_negative_sizes) {
  boost::mpi::communicator world;
  int m_a = -3;
  int n_a = 3;
  int m_b = 3;
  int n_b = 3;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(m_a);
    taskDataPar->inputs_count.emplace_back(n_a);
    taskDataPar->inputs_count.emplace_back(m_b);
    taskDataPar->inputs_count.emplace_back(n_b);
  }

  smirnov_i_tape_splitting_A::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
}

TEST(smirnov_i_tape_splitting_A_mpi, invalid_matrix_size) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    int m_a = 0;
    int n_a = 10;
    std::unique_ptr<double[]> A(new double[m_a * n_a]);
    ASSERT_ANY_THROW(smirnov_i_tape_splitting_A::get_random_matrix(A.get(), m_a * n_a));
  }
}
TEST(smirnov_i_tape_splitting_A_mpi, cant_mult_matrix_wrong_sizes) {
  boost::mpi::communicator world;
  double* A = nullptr;
  double* B = nullptr;
  double* res = nullptr;
  int m_a = 2;
  int n_a = 3;
  int m_b = 7;
  int n_b = 4;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    A = new double[m_a * n_a];
    B = new double[m_b * n_b];
    smirnov_i_tape_splitting_A::get_random_matrix(A, m_a * n_a);
    smirnov_i_tape_splitting_A::get_random_matrix(B, m_b * n_b);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));

    taskDataPar->inputs_count.emplace_back(m_a);
    taskDataPar->inputs_count.emplace_back(n_a);
    taskDataPar->inputs_count.emplace_back(m_b);
    taskDataPar->inputs_count.emplace_back(n_b);
    res = new double[m_a * n_b];
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res));
    taskDataPar->outputs_count.emplace_back(m_a);
    taskDataPar->outputs_count.emplace_back(n_b);
  }
  smirnov_i_tape_splitting_A::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
  if(world.rank() == 0) {
    delete[] A;
    delete[] B;
    delete[] res;
  }
}

TEST(smirnov_i_tape_splitting_A_mpi, mult_matrix_and_vector) {
  boost::mpi::communicator world;

  double* A = nullptr;
  double* B = nullptr;
  double* res = nullptr;
  int m_a = 70;
  int n_a = 50;
  int m_b = 50;
  int n_b = 1;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    A = new double[m_a * n_a];
    B = new double[m_b * n_b];
    smirnov_i_tape_splitting_A::get_random_matrix(A, m_a * n_a);
    smirnov_i_tape_splitting_A::get_random_matrix(B, m_b * n_b);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));

    taskDataPar->inputs_count.emplace_back(m_a);
    taskDataPar->inputs_count.emplace_back(n_a);
    taskDataPar->inputs_count.emplace_back(m_b);
    taskDataPar->inputs_count.emplace_back(n_b);

    res = new double[m_a * n_b];
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res));
    taskDataPar->outputs_count.emplace_back(m_a);
    taskDataPar->outputs_count.emplace_back(n_b);
  }

  smirnov_i_tape_splitting_A::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));

    taskDataSeq->inputs_count.emplace_back(m_a);
    taskDataSeq->inputs_count.emplace_back(n_a);
    taskDataSeq->inputs_count.emplace_back(m_b);
    taskDataSeq->inputs_count.emplace_back(n_b);

    auto* res_seq = new double[m_a * n_b];
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_seq));
    taskDataSeq->outputs_count.emplace_back(m_a);
    taskDataSeq->outputs_count.emplace_back(n_b);

    auto TestTaskSequential = std::make_shared<smirnov_i_tape_splitting_A::TestMPITaskSequential>(taskDataSeq);
    ASSERT_EQ(TestTaskSequential->validation(), true);
    TestTaskSequential->pre_processing();
    TestTaskSequential->run();
    TestTaskSequential->post_processing();

    for (int i = 0; i < m_a * n_b; i++) {
      ASSERT_NEAR(res_seq[i], res[i], 1e-6);
    }

    delete[] res_seq;
  }
  if (world.rank() == 0) {
    delete[] A;
    delete[] B;
    delete[] res;
  }
}

TEST(smirnov_i_tape_splitting_A_mpi, mult_matrix_and_matrix_small) {
  boost::mpi::communicator world;

  double* A = nullptr;
  double* B = nullptr;
  double* res = nullptr;
  int m_a = 2;
  int n_a = 3;
  int m_b = 3;
  int n_b = 4;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    A = new double[m_a * n_a];
    B = new double[m_b * n_b];
    smirnov_i_tape_splitting_A::get_random_matrix(A, m_a * n_a);
    smirnov_i_tape_splitting_A::get_random_matrix(B, m_b * n_b);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));

    taskDataPar->inputs_count.emplace_back(m_a);
    taskDataPar->inputs_count.emplace_back(n_a);
    taskDataPar->inputs_count.emplace_back(m_b);
    taskDataPar->inputs_count.emplace_back(n_b);

    res = new double[m_a * n_b];
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res));
    taskDataPar->outputs_count.emplace_back(m_a);
    taskDataPar->outputs_count.emplace_back(n_b);
  }

  smirnov_i_tape_splitting_A::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));

    taskDataSeq->inputs_count.emplace_back(m_a);
    taskDataSeq->inputs_count.emplace_back(n_a);
    taskDataSeq->inputs_count.emplace_back(m_b);
    taskDataSeq->inputs_count.emplace_back(n_b);

    auto* res_seq = new double[m_a * n_b];
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_seq));
    taskDataSeq->outputs_count.emplace_back(m_a);
    taskDataSeq->outputs_count.emplace_back(n_b);

    auto TestTaskSequential = std::make_shared<smirnov_i_tape_splitting_A::TestMPITaskSequential>(taskDataSeq);

    ASSERT_EQ(TestTaskSequential->validation(), true);
    TestTaskSequential->pre_processing();
    TestTaskSequential->run();
    TestTaskSequential->post_processing();

    for (int i = 0; i < m_a * n_b; i++) {
      ASSERT_NEAR(res_seq[i], res[i], 1e-6);
    }
    delete[] res_seq;
  }
  if (world.rank() == 0) {
    delete[] A;
    delete[] B;
    delete[] res;
  }
}

TEST(smirnov_i_tape_splitting_A_mpi, mult_matrix_and_matrix_medium) {
  boost::mpi::communicator world;

  double* A = nullptr;
  double* B = nullptr;
  double* res = nullptr;
  int m_a = 200;
  int n_a = 300;
  int m_b = 300;
  int n_b = 400;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    A = new double[m_a * n_a];
    B = new double[m_b * n_b];
    smirnov_i_tape_splitting_A::get_random_matrix(A, m_a * n_a);
    smirnov_i_tape_splitting_A::get_random_matrix(B, m_b * n_b);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));

    taskDataPar->inputs_count.emplace_back(m_a);
    taskDataPar->inputs_count.emplace_back(n_a);
    taskDataPar->inputs_count.emplace_back(m_b);
    taskDataPar->inputs_count.emplace_back(n_b);

    res = new double[m_a * n_b];
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res));
    taskDataPar->outputs_count.emplace_back(m_a);
    taskDataPar->outputs_count.emplace_back(n_b);
  }

  smirnov_i_tape_splitting_A::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));

    taskDataSeq->inputs_count.emplace_back(m_a);
    taskDataSeq->inputs_count.emplace_back(n_a);
    taskDataSeq->inputs_count.emplace_back(m_b);
    taskDataSeq->inputs_count.emplace_back(n_b);

    auto* res_seq = new double[m_a * n_b];
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_seq));
    taskDataSeq->outputs_count.emplace_back(m_a);
    taskDataSeq->outputs_count.emplace_back(n_b);

    auto TestTaskSequential = std::make_shared<smirnov_i_tape_splitting_A::TestMPITaskSequential>(taskDataSeq);

    ASSERT_EQ(TestTaskSequential->validation(), true);
    TestTaskSequential->pre_processing();
    TestTaskSequential->run();
    TestTaskSequential->post_processing();

    for (int i = 0; i < m_a * n_b; i++) {
      ASSERT_NEAR(res_seq[i], res[i], 1e-6);
    }

    delete[] res_seq;
  }
  if (world.rank() == 0) {
    delete[] A;
    delete[] B;
    delete[] res;
  }
}

TEST(smirnov_i_tape_splitting_A_mpi, mult_matrix_and_matrix_large) {
  boost::mpi::communicator world;

  double* A = nullptr;
  double* B = nullptr;
  double* res = nullptr;
  int m_a = 700;
  int n_a = 800;
  int m_b = 800;
  int n_b = 100;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    A = new double[m_a * n_a];
    B = new double[m_b * n_b];
    smirnov_i_tape_splitting_A::get_random_matrix(A, m_a * n_a);
    smirnov_i_tape_splitting_A::get_random_matrix(B, m_b * n_b);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));

    taskDataPar->inputs_count.emplace_back(m_a);
    taskDataPar->inputs_count.emplace_back(n_a);
    taskDataPar->inputs_count.emplace_back(m_b);
    taskDataPar->inputs_count.emplace_back(n_b);

    res = new double[m_a * n_b];
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res));
    taskDataPar->outputs_count.emplace_back(m_a);
    taskDataPar->outputs_count.emplace_back(n_b);
  }

  smirnov_i_tape_splitting_A::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));

    taskDataSeq->inputs_count.emplace_back(m_a);
    taskDataSeq->inputs_count.emplace_back(n_a);
    taskDataSeq->inputs_count.emplace_back(m_b);
    taskDataSeq->inputs_count.emplace_back(n_b);

    auto* res_seq = new double[m_a * n_b];
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_seq));
    taskDataSeq->outputs_count.emplace_back(m_a);
    taskDataSeq->outputs_count.emplace_back(n_b);

    auto TestTaskSequential = std::make_shared<smirnov_i_tape_splitting_A::TestMPITaskSequential>(taskDataSeq);

    ASSERT_EQ(TestTaskSequential->validation(), true);
    TestTaskSequential->pre_processing();
    TestTaskSequential->run();
    TestTaskSequential->post_processing();

    for (int i = 0; i < m_a * n_b; i++) {
      ASSERT_NEAR(res_seq[i], res[i], 1e-6);
    }

    delete[] res_seq;
  }
  if (world.rank() == 0) {
    delete[] A;
    delete[] B;
    delete[] res;
  }
}