// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/malyshev_a_simple_iteration_method/include/ops_mpi.hpp"

TEST(malyshev_a_simple_iteration_method_mpi, basic_test) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  std::vector<double> A{3, 2, -1, 1, -2, 1, 2, -3, -5};
  std::vector<double> X(3, 0);
  std::vector<double> B{8, -2, 1};
  std::vector<double> X0(3, 0);
  double eps = 1e-4;
  if (world.rank() == 0) {
    // Create data

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
    taskDataPar->inputs_count.emplace_back(X.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->outputs_count.emplace_back(X.size());
  }
  // Create Task
  malyshev_a_simple_iteration_method_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> X_seq(3, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
    taskDataSeq->inputs_count.emplace_back(X_seq.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(X_seq.data()));
    taskDataSeq->outputs_count.emplace_back(X_seq.size());

    malyshev_a_simple_iteration_method_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    double sum_mpi_eq;
    double sum_seq_eq;
    for (uint32_t i = 0; i < X.size(); i++) {
      sum_mpi_eq = 0;
      sum_seq_eq = 0;

      for (uint32_t j = 0; j < X.size(); j++) {
        sum_mpi_eq += X[j] * A[i * X.size() + j];
        sum_seq_eq += X_seq[j] * A[i * X.size() + j];
      }

      ASSERT_TRUE(std::abs(sum_mpi_eq - B[i]) <= eps);
      ASSERT_TRUE(std::abs(sum_seq_eq - B[i]) <= eps);
      ASSERT_TRUE(std::abs(sum_seq_eq - sum_mpi_eq) <= eps);
    }
  }
}

TEST(malyshev_a_simple_iteration_method_mpi, random_test) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int size = 10;
  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> X;
  std::vector<double> X0(size, 0);
  double eps = 1e-4;
  if (world.rank() == 0) {
    // Create data
    X.resize(size);
    malyshev_a_simple_iteration_method_mpi::getRandomData(size, A, B);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
    taskDataPar->inputs_count.emplace_back(X.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->outputs_count.emplace_back(X.size());
  }
  // Create Task
  malyshev_a_simple_iteration_method_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> X_seq(size, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
    taskDataSeq->inputs_count.emplace_back(X_seq.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(X_seq.data()));
    taskDataSeq->outputs_count.emplace_back(X_seq.size());

    malyshev_a_simple_iteration_method_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    double sum_mpi_eq;
    double sum_seq_eq;
    for (uint32_t i = 0; i < X.size(); i++) {
      sum_mpi_eq = 0;
      sum_seq_eq = 0;

      for (uint32_t j = 0; j < X.size(); j++) {
        sum_mpi_eq += X[j] * A[i * X.size() + j];
        sum_seq_eq += X_seq[j] * A[i * X.size() + j];
      }

      ASSERT_TRUE(std::abs(sum_mpi_eq - B[i]) <= eps);
      ASSERT_TRUE(std::abs(sum_seq_eq - B[i]) <= eps);
      ASSERT_TRUE(std::abs(sum_seq_eq - sum_mpi_eq) <= eps);
    }
  }
}

TEST(malyshev_a_simple_iteration_method_mpi, validate_input_data) {
  boost::mpi::communicator world;

  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> X;
  std::vector<double> X0;
  double eps = 1e-4;

  const auto try_validate = [](auto &taskData) {
    malyshev_a_simple_iteration_method_mpi::TestMPITaskParallel testTaskParallel(taskData);
    return testTaskParallel.validation();
  };

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    ASSERT_FALSE(try_validate(taskDataPar));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    ASSERT_FALSE(try_validate(taskDataPar));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    ASSERT_FALSE(try_validate(taskDataPar));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
    ASSERT_FALSE(try_validate(taskDataPar));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    ASSERT_FALSE(try_validate(taskDataPar));
    taskDataPar->inputs_count.emplace_back(X.size());
    ASSERT_FALSE(try_validate(taskDataPar));
  }
}

TEST(malyshev_a_simple_iteration_method_mpi, validate_input_determinant) {
  boost::mpi::communicator world;

  std::vector<double> A{3, 0, -1, 1, 0, 1, 2, 0, -5};
  std::vector<double> X(3, 0);
  std::vector<double> B{8, -2, 1};
  std::vector<double> X0(3, 0);
  double eps = 1e-4;

  const auto try_validate = [](auto &taskData) {
    malyshev_a_simple_iteration_method_mpi::TestMPITaskParallel testTaskParallel(taskData);
    return testTaskParallel.validation();
  };

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
    taskDataPar->inputs_count.emplace_back(X.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->outputs_count.emplace_back(X.size());

    ASSERT_FALSE(try_validate(taskDataPar));
  }
}

TEST(malyshev_a_simple_iteration_method_mpi, validate_input_rank) {
  boost::mpi::communicator world;

  std::vector<double> A{1, 1, 3, 3};
  std::vector<double> X(2, 0);
  std::vector<double> B{1, 2};
  std::vector<double> X0(2, 0);
  double eps = 1e-4;

  const auto try_validate = [](auto &taskData) {
    malyshev_a_simple_iteration_method_mpi::TestMPITaskParallel testTaskParallel(taskData);
    return testTaskParallel.validation();
  };

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
    taskDataPar->inputs_count.emplace_back(X.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->outputs_count.emplace_back(X.size());

    ASSERT_FALSE(try_validate(taskDataPar));
  }
}

TEST(malyshev_a_simple_iteration_method_mpi, validate_input_slowly_converging) {
  boost::mpi::communicator world;

  std::vector<double> A{5, -7, 3, 2};
  std::vector<double> X(2, 0);
  std::vector<double> B{1, 2};
  std::vector<double> X0(2, 0);
  double eps = 1e-4;

  const auto try_validate = [](auto &taskData) {
    malyshev_a_simple_iteration_method_mpi::TestMPITaskParallel testTaskParallel(taskData);
    return testTaskParallel.validation();
  };

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
    taskDataPar->inputs_count.emplace_back(X.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->outputs_count.emplace_back(X.size());

    ASSERT_FALSE(try_validate(taskDataPar));
  }
}

TEST(malyshev_a_simple_iteration_method_mpi, validate_input_zero_on_the_main_diagonal) {
  boost::mpi::communicator world;

  std::vector<double> A{0, 4, 2, -3, 0, 4, 6, 1, 0};
  std::vector<double> X(3, 0);
  std::vector<double> B{1, 2, 1};
  std::vector<double> X0(3, 0);
  double eps = 1e-4;

  const auto try_validate = [](auto &taskData) {
    malyshev_a_simple_iteration_method_mpi::TestMPITaskParallel testTaskParallel(taskData);
    return testTaskParallel.validation();
  };

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
    taskDataPar->inputs_count.emplace_back(X.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->outputs_count.emplace_back(X.size());

    // We expect false because the system is slowly converging
    ASSERT_FALSE(try_validate(taskDataPar));
  }
}