#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/zolotareva_a_SLE_gradient_method/include/ops_mpi.hpp"

namespace zolotareva_a_SLE_gradient_method_mpi {
void generateSLE(std::vector<double>& A, std::vector<double>& b, int n, double min, double max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(min, max);

  for (int i = 0; i < n; ++i) {
    b[i] = dist(gen);
    for (int j = i; j < n; ++j) {
      double value = dist(gen);
      A[i * n + j] = value;
      A[j * n + i] = value;  // Обеспечение симметричности
    }
  }

  for (int i = 0; i < n; ++i) {
    A[i * n + i] += n * max;  // Обеспечение доминирования диагонали
  }
}

void form(int n_) {
  boost::mpi::communicator world;
  int n = n_;
  std::vector<double> A(n * n);
  std::vector<double> b(n);
  std::vector<double> mpi_x(n);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    generateSLE(A, b, n, -100.0f, 100.0f);
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataPar->inputs_count.push_back(n * n);
    taskDataPar->inputs_count.push_back(n);
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(mpi_x.data()));
    taskDataPar->outputs_count.push_back(n);
  }

  zolotareva_a_SLE_gradient_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> seq_x(n_);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataSeq->inputs_count.push_back(n * n);
    taskDataSeq->inputs_count.push_back(n);
    taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t*>(seq_x.data()));
    taskDataSeq->outputs_count.push_back(n);

    zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (int i = 0; i < n; ++i) {
      EXPECT_NEAR(seq_x[i], mpi_x[i], 1e-10);
    }
  }
}
}  // namespace zolotareva_a_SLE_gradient_method_mpi

TEST(zolotareva_a_SLE_gradient_method_mpi, validation_non_symmetric) {
  boost::mpi::communicator world;
  int n = 2;
  std::vector<double> A = {1, 2, 3, 4};  // не симметричная
  std::vector<double> b = {5, 6};
  std::vector<double> x(n);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
    taskData->inputs_count.push_back(n * n);
    taskData->inputs_count.push_back(n);
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
    taskData->outputs_count.push_back(n);
  }

  zolotareva_a_SLE_gradient_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskData);
  if (world.rank() == 0) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
  }
}

TEST(zolotareva_a_SLE_gradient_method_mpi, small_system_n_1) {
  boost::mpi::communicator world;
  int n = 1;
  std::vector<double> A;
  std::vector<double> b;
  std::vector<double> x(n);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    A = {2};
    b = {4};
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
    taskData->inputs_count.push_back(n * n);
    taskData->inputs_count.push_back(n);
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
    taskData->outputs_count.push_back(n);
  }

  zolotareva_a_SLE_gradient_method_mpi::TestMPITaskParallel task(taskData);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  if (world.rank() == 0) {
    EXPECT_NEAR(x[0], 2.0, 1e-8);
  }
}
TEST(zolotareva_a_SLE_gradient_method_mpi, system_not_positive_definite) {
  boost::mpi::communicator world;
  int n = 2;
  std::vector<double> A;
  std::vector<double> b;
  std::vector<double> x(n);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    A = {0, 0, 0, 0};
    b = {0, 0};
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
    taskData->inputs_count.push_back(n * n);
    taskData->inputs_count.push_back(n);
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
    taskData->outputs_count.push_back(n);
  }

  zolotareva_a_SLE_gradient_method_mpi::TestMPITaskParallel task(taskData);
  if (world.rank() == 0) {
    ASSERT_FALSE(task.validation());
  }
}
TEST(zolotareva_a_SLE_gradient_method_mpi, zero_dimension) {
  boost::mpi::communicator world;
  std::vector<double> A;
  std::vector<double> b;
  std::vector<double> x;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs_count.push_back(0);
    taskData->inputs_count.push_back(0);
    taskData->outputs_count.push_back(0);
  }

  zolotareva_a_SLE_gradient_method_mpi::TestMPITaskParallel task(taskData);
  if (world.rank() == 0) {
    ASSERT_FALSE(task.validation());
  }
}
TEST(zolotareva_a_SLE_gradient_method_mpi, system_n_3_random) { zolotareva_a_SLE_gradient_method_mpi::form(3); }
TEST(zolotareva_a_SLE_gradient_method_mpi, system_n_4_random) { zolotareva_a_SLE_gradient_method_mpi::form(4); }
TEST(zolotareva_a_SLE_gradient_method_mpi, system_n_5_random) { zolotareva_a_SLE_gradient_method_mpi::form(5); }
TEST(zolotareva_a_SLE_gradient_method_mpi, system_n_17_random) { zolotareva_a_SLE_gradient_method_mpi::form(17); }
TEST(zolotareva_a_SLE_gradient_method_mpi, system_n_51_random) { zolotareva_a_SLE_gradient_method_mpi::form(51); }
TEST(zolotareva_a_SLE_gradient_method_mpi, system_n_123_random) { zolotareva_a_SLE_gradient_method_mpi::form(123); }
TEST(zolotareva_a_SLE_gradient_method_mpi, system_n_591_random) { zolotareva_a_SLE_gradient_method_mpi::form(591); }
