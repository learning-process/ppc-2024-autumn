#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "mpi/Sdobnov_V_iteration_method_yakoby/include/ops_mpi.hpp"

std::pair<std::vector<double>, std::vector<double>> generate_correct_matrix(int n, double min_val = -10.0,
                                                                            double max_val = 10.0) {
  std::vector<double> A(n * n);
  std::vector<double> b(n);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(min_val, max_val);
  for (int i = 0; i < n; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < n; ++j) {
      if (i != j) {
        A[i * n + j] = dist(gen);
        row_sum += std::abs(A[i * n + j]);
      }
    }
    A[i * n + i] = row_sum + std::abs(dist(gen)) + 1.0;
    b[i] = dist(gen);
  }
  return {A, b};
}

TEST(Sdobnov_V_iteration_method_yakoby_par, InvalidMatrixWithoutDiagonalDominance) {
  boost::mpi::communicator world;
  int size = 3;
  std::vector<double> matrix = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0};
  std::vector<double> free_members = {5.0, 5.0, 5.0};
  std::vector<double> res(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(free_members.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  }

  Sdobnov_iteration_method_yakoby_par::IterationMethodYakobyPar test(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_FALSE(test.validation());
  }
}

TEST(Sdobnov_V_iteration_method_yakoby_par, InvalidInputCount) {
  boost::mpi::communicator world;
  int size = 3;
  std::vector<double> matrix = {2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 3.0};
  std::vector<double> free_members = {5.0, 5.0, 5.0};
  std::vector<double> res(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(free_members.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  }

  Sdobnov_iteration_method_yakoby_par::IterationMethodYakobyPar test(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_FALSE(test.validation());
  }
}

TEST(Sdobnov_V_iteration_method_yakoby_par, InvalidInput) {
  boost::mpi::communicator world;
  int size = 3;
  std::vector<double> matrix = {2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 3.0};
  std::vector<double> free_members = {5.0, 5.0, 5.0};
  std::vector<double> res(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(-size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  }

  Sdobnov_iteration_method_yakoby_par::IterationMethodYakobyPar test(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_FALSE(test.validation());
  }
}

TEST(Sdobnov_V_iteration_method_yakoby_par, InvalidOutputCount) {
  boost::mpi::communicator world;
  int size = 3;
  std::vector<double> matrix = {2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 3.0};
  std::vector<double> free_members = {5.0, 5.0, 5.0};
  std::vector<double> res(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(-size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(free_members.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  }

  Sdobnov_iteration_method_yakoby_par::IterationMethodYakobyPar test(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_FALSE(test.validation());
  }
}

TEST(Sdobnov_V_iteration_method_yakoby_par, InvalidOutput) {
  boost::mpi::communicator world;
  int size = 3;
  std::vector<double> matrix = {2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 3.0};
  std::vector<double> free_members = {5.0, 5.0, 5.0};
  std::vector<double> res(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(-size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->outputs_count.emplace_back(size);
  }
  Sdobnov_iteration_method_yakoby_par::IterationMethodYakobyPar test(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_FALSE(test.validation());
  }
}

TEST(Sdobnov_V_iteration_method_yakoby_par, IterationMethodTest3x3) {
  boost::mpi::communicator world;
  int size = 3;
  std::vector<double> respar(size, 0.0);
  std::vector<double> input_matrix = {2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 3.0};
  std::vector<double> input_free_members = {6.0, 6.0, 6.0};
  std::vector<double> expected_res = {2.0, 2.0, 2.0};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_free_members.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(respar.data()));
  }

  Sdobnov_iteration_method_yakoby_par::IterationMethodYakobyPar test(taskDataPar);

  test.validation();
  test.pre_processing();
  test.run();
  test.post_processing();

  if (world.rank() == 0) {
    std::vector<double> resseq(size, 0.0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_free_members.data()));
    taskDataSeq->outputs_count.emplace_back(size);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resseq.data()));

    Sdobnov_iteration_method_yakoby_par::IterationMethodYakobySeq testseq(taskDataSeq);

    testseq.validation();
    testseq.pre_processing();
    testseq.run();
    testseq.post_processing();

    for (int i = 0; i < size; i++) {
      ASSERT_NEAR(respar[i], resseq[i], 1e-3);
    }
  }
}

TEST(Sdobnov_V_iteration_method_yakoby_par, IterationMethodTest4x4) {
  boost::mpi::communicator world;
  int size = 4;
  std::vector<double> respar(size, 0.0);
  std::vector<double> input_matrix = {12.0, 4.0, -2.0, 0.0, 2.0, 9.0,  3.0,  -3.0,
                                      -2.0, 1.0, 6.0,  2.0, 0.0, -1.0, -7.0, 10.0};
  std::vector<double> input_free_members = {10.0, 9.0, 8.0, 7.0};
  std::vector<double> expected_res = {0.637086, 1.036719, 0.895960, 1.430844};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_free_members.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(respar.data()));
  }

  Sdobnov_iteration_method_yakoby_par::IterationMethodYakobyPar test(taskDataPar);

  test.validation();
  test.pre_processing();
  test.run();
  test.post_processing();

  if (world.rank() == 0) {
    std::vector<double> resseq(size, 0.0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_free_members.data()));
    taskDataSeq->outputs_count.emplace_back(size);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resseq.data()));

    Sdobnov_iteration_method_yakoby_par::IterationMethodYakobySeq testseq(taskDataSeq);

    testseq.validation();
    testseq.pre_processing();
    testseq.run();
    testseq.post_processing();

    for (int i = 0; i < size; i++) {
      ASSERT_NEAR(respar[i], resseq[i], 1e-3);
    }
  }
}

TEST(Sdobnov_V_iteration_method_yakoby_par, RandomMatrix10x10_1) {
  boost::mpi::communicator world;
  int size = 10;
  auto [input_matrix, input_free_members] = generate_correct_matrix(size);
  std::vector<double> respar(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_free_members.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(respar.data()));
  }

  Sdobnov_iteration_method_yakoby_par::IterationMethodYakobyPar test(taskDataPar);

  test.validation();
  test.pre_processing();
  test.run();
  test.post_processing();

  if (world.rank() == 0) {
    std::vector<double> resseq(size, 0.0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_free_members.data()));
    taskDataSeq->outputs_count.emplace_back(size);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resseq.data()));

    Sdobnov_iteration_method_yakoby_par::IterationMethodYakobySeq testseq(taskDataSeq);

    testseq.validation();
    testseq.pre_processing();
    testseq.run();
    testseq.post_processing();

    for (int i = 0; i < size; i++) {
      ASSERT_NEAR(respar[i], resseq[i], 1e-3);
    }
  }
}

TEST(Sdobnov_V_iteration_method_yakoby_par, RandomMatrix10x10_2) {
  boost::mpi::communicator world;
  int size = 10;
  auto [input_matrix, input_free_members] = generate_correct_matrix(size);
  std::vector<double> respar(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_free_members.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(respar.data()));
  }

  Sdobnov_iteration_method_yakoby_par::IterationMethodYakobyPar test(taskDataPar);

  test.validation();
  test.pre_processing();
  test.run();
  test.post_processing();

  if (world.rank() == 0) {
    std::vector<double> resseq(size, 0.0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_free_members.data()));
    taskDataSeq->outputs_count.emplace_back(size);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resseq.data()));

    Sdobnov_iteration_method_yakoby_par::IterationMethodYakobySeq testseq(taskDataSeq);

    testseq.validation();
    testseq.pre_processing();
    testseq.run();
    testseq.post_processing();

    for (int i = 0; i < size; i++) {
      ASSERT_NEAR(respar[i], resseq[i], 1e-3);
    }
  }
}

TEST(Sdobnov_V_iteration_method_yakoby_par, RandomMatrix10x10_3) {
  boost::mpi::communicator world;
  int size = 10;
  auto [input_matrix, input_free_members] = generate_correct_matrix(size);
  std::vector<double> respar(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_free_members.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(respar.data()));
  }

  Sdobnov_iteration_method_yakoby_par::IterationMethodYakobyPar test(taskDataPar);

  test.validation();
  test.pre_processing();
  test.run();
  test.post_processing();

  if (world.rank() == 0) {
    std::vector<double> resseq(size, 0.0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_free_members.data()));
    taskDataSeq->outputs_count.emplace_back(size);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resseq.data()));

    Sdobnov_iteration_method_yakoby_par::IterationMethodYakobySeq testseq(taskDataSeq);

    testseq.validation();
    testseq.pre_processing();
    testseq.run();
    testseq.post_processing();

    for (int i = 0; i < size; i++) {
      ASSERT_NEAR(respar[i], resseq[i], 1e-3);
    }
  }
}