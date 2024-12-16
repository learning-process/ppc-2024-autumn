// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/zolotareva_a_SLE_gradient_method/include/ops_mpi.hpp"
using namespace std;
namespace zolotareva_a_SLE_gradient_method_mpi {
void generateSLE(std::vector<double>& A, std::vector<double>& b, int n) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-10.0f, 10.0f);

  for (int i = 0; i < n; ++i) {
    b[i] = dist(gen);
    for (int j = i; j < n; ++j) {
      double value = dist(gen);
      A[i * n + j] = value;
      A[j * n + i] = value;  // Обеспечение симметричности
    }
  }

  for (int i = 0; i < n; ++i) {
    A[i * n + i] += n * 10.0f;  // Обеспечение доминирования диагонали
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
    generateSLE(A, b, n);
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataPar->inputs_count.push_back(n * n);
    taskDataPar->inputs_count.push_back(n);
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(mpi_x.data()));
    taskDataPar->outputs_count.push_back(mpi_x.size());
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
    taskDataSeq->outputs_count.push_back(seq_x.size());

    zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    EXPECT_EQ(seq_x, mpi_x);
  }
}
}  // namespace zolotareva_a_SLE_gradient_method_mpi

TEST(zolotareva_a_SLE_gradient_method_mpi, Test_random_3x3) { zolotareva_a_SLE_gradient_method_mpi::form(3); }
