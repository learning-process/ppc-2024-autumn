// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <random>
#include <vector>

#include "mpi/rezantseva_a_simple_iteration_method/include/ops_mpi_rezantseva.hpp"

static int offset = 0;

std::pair<std::vector<double>, std::vector<double>> rezantseva_a_simple_iteration_method_mpi::createRandomMatrix(
    size_t n) {
  std::vector<double> A(n * n);
  std::vector<double> b(n);
  std::mt19937 gen;
  gen.seed((unsigned)time(nullptr) + ++offset);

  for (size_t i = 0; i < n; i++) {
    double sum = 0.0;

    // gen non diagonal elements
    for (size_t j = 0; j < n; j++) {
      if (i != j) {
        A[i * n + j] = static_cast<double>(gen() % 50 - 25);  // from -25 to 24
        sum += std::abs(A[i * n + j]);
      }
    }
    A[i * n + i] = sum + static_cast<double>(gen() % 50 + 1);
    b[i] = static_cast<double>(gen() % 100);  // from 0 to 99
  }
  return {A, b};
}

TEST(rezantseva_a_simple_iteration_method_mpi, check_matrix_10x10) {
  boost::mpi::communicator world;
  size_t size = 10;
  auto [A, b] = rezantseva_a_simple_iteration_method_mpi::createRandomMatrix(size);
  std::vector<double> out(size, 0.0);
  const double eps = 1e-3;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(b.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  // Create Task
  rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_res(size, 0.0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(A.size());
    taskDataSeq->inputs_count.emplace_back(b.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // Create Task
    rezantseva_a_simple_iteration_method_mpi::SimpleIterationSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    for (size_t i = 0; i < size; ++i) {
      ASSERT_NEAR(reference_res[i], out[i], eps);
    }
  }
}

TEST(rezantseva_a_simple_iteration_method_mpi, check_matrix_100x100) {
  boost::mpi::communicator world;
  size_t size = 100;
  auto [A, b] = rezantseva_a_simple_iteration_method_mpi::createRandomMatrix(size);
  std::vector<double> out(size, 0.0);
  const double eps = 1e-3;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(b.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  // Create Task
  rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_res(size, 0.0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(A.size());
    taskDataSeq->inputs_count.emplace_back(b.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // Create Task
    rezantseva_a_simple_iteration_method_mpi::SimpleIterationSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    for (size_t i = 0; i < size; ++i) {
      ASSERT_NEAR(reference_res[i], out[i], eps);
    }
  }
}

TEST(rezantseva_a_simple_iteration_method_mpi, check_matrix_1000x1000) {
  boost::mpi::communicator world;
  size_t size = 1000;
  auto [A, b] = rezantseva_a_simple_iteration_method_mpi::createRandomMatrix(size);
  std::vector<double> out(size, 0.0);
  const double eps = 1e-3;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(b.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  // Create Task
  rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_res(size, 0.0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(A.size());
    taskDataSeq->inputs_count.emplace_back(b.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // Create Task
    rezantseva_a_simple_iteration_method_mpi::SimpleIterationSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    for (size_t i = 0; i < size; ++i) {
      ASSERT_NEAR(reference_res[i], out[i], eps);
    }
  }
}

TEST(rezantseva_a_simple_iteration_method_mpi, check_matrix_3x3) {
  boost::mpi::communicator world;
  size_t size = 3;
  const double eps = 1e-3;
  // Create data
  std::vector<double> A = {4.0, 1.0, 2.0, 5.0, 9.0, 3.0, 4.0, 2.0, 12.0};
  std::vector<double> b = {1.0, 2.0, 1.0};
  std::vector<double> answer = {0.23, 0.0971, -0.00944};
  std::vector<double> out(size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(b.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  // Create Task
  rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (size_t i = 0; i < size; ++i) {
      ASSERT_NEAR(out[i], answer[i], eps);
    }
  }

  if (world.rank() == 0) {
    std::vector<double> reference_res(size, 0.0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(A.size());
    taskDataSeq->inputs_count.emplace_back(b.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // Create Task
    rezantseva_a_simple_iteration_method_mpi::SimpleIterationSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    for (size_t i = 0; i < size; ++i) {
      ASSERT_NEAR(reference_res[i], answer[i], eps);
    }
  }
}

TEST(rezantseva_a_simple_iteration_method_mpi, check_matrix_4x4) {
  boost::mpi::communicator world;
  size_t size = 4;
  const double eps = 1e-3;
  // Create data
  std::vector<double> A = {5.0, 1.0, 1.0, 1.0, 1.0, 7.0, 1.0, 1.0, 1.0, 1.0, 6.0, 1.0, 1.0, 1.0, 1.0, 4.0};
  std::vector<double> b = {8.0, 10.0, 9.0, 7.0};
  std::vector<double> answer = {1.0, 1.0, 1.0, 1.0};
  std::vector<double> out(size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(b.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  // Create Task
  rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (size_t i = 0; i < size; ++i) {
      ASSERT_NEAR(out[i], answer[i], eps);
    }
  }

  if (world.rank() == 0) {
    std::vector<double> reference_res(size, 0.0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(A.size());
    taskDataSeq->inputs_count.emplace_back(b.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // Create Task
    rezantseva_a_simple_iteration_method_mpi::SimpleIterationSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    for (size_t i = 0; i < size; ++i) {
      ASSERT_NEAR(reference_res[i], answer[i], eps);
    }
  }
}

TEST(rezantseva_a_simple_iteration_method_mpi, check_matrix_5x5) {
  boost::mpi::communicator world;
  size_t size = 5;
  const double eps = 1e-3;
  // Create data
  std::vector<double> A = {10.0, -2.0, -1.0, -1.0, -1.0, -2.0, 12.0, -3.0, -1.0, -1.0, -1.0, -3.0, 15.0,
                           -1.0, -1.0, -1.0, -1.0, -1.0, 11.0, -2.0, -1.0, -1.0, -1.0, -2.0, 14.0};
  std::vector<double> b = {8.0, 10.0, 9.0, 7.0, 11.0};
  std::vector<double> answer = {1.488, 1.587, 1.185, 1.254, 1.269};
  std::vector<double> out(size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(b.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  // Create Task
  rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (size_t i = 0; i < size; ++i) {
      ASSERT_NEAR(out[i], answer[i], eps);
    }
  }

  if (world.rank() == 0) {
    std::vector<double> reference_res(size, 0.0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(A.size());
    taskDataSeq->inputs_count.emplace_back(b.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // Create Task
    rezantseva_a_simple_iteration_method_mpi::SimpleIterationSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    for (size_t i = 0; i < size; ++i) {
      ASSERT_NEAR(reference_res[i], answer[i], eps);
    }
  }
}

TEST(rezantseva_a_simple_iteration_method_mpi, check_matrix_6x6) {
  boost::mpi::communicator world;
  size_t size = 6;
  const double eps = 1e-3;
  // Create data
  std::vector<double> A = {10.0, -2.0, -1.0, -1.0, -1.0, -3.0, -2.0, 12.0, -3.0, -1.0, -1.0, -3.0,
                           -1.0, -3.0, 15.0, -1.0, -1.0, -2.0, -1.0, -1.0, -1.0, 11.0, -2.0, -3.0,
                           -1.0, -1.0, -1.0, -2.0, 14.0, -5.0, -1.0, -2.0, -1.0, -1.0, -3.0, 17};
  std::vector<double> b = {8.0, 10.0, 9.0, 7.0, 11.0, 5.0};
  std::vector<double> answer = {2.144, 2.209, 1.598, 1.857, 1.912, 1.22};
  std::vector<double> out(size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(b.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  // Create Task
  rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (size_t i = 0; i < size; ++i) {
      ASSERT_NEAR(out[i], answer[i], eps);
    }
  }

  if (world.rank() == 0) {
    std::vector<double> reference_res(size, 0.0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(A.size());
    taskDataSeq->inputs_count.emplace_back(b.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // Create Task
    rezantseva_a_simple_iteration_method_mpi::SimpleIterationSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    for (size_t i = 0; i < size; ++i) {
      ASSERT_NEAR(reference_res[i], answer[i], eps);
    }
  }
}

TEST(rezantseva_a_simple_iteration_method_mpi, check_validation_invalid_inputs_count) {
  boost::mpi::communicator world;
  size_t size = 3;
  // Create data
  std::vector<double> A = {4.0, 1.0, 2.0, 5.0, 9.0, 3.0, 4.0, 2.0, 12.0};
  std::vector<double> b = {1.0, 2.0, 1.0};
  std::vector<double> out(size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(b.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  // Create Task
  rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), false);
}

TEST(rezantseva_a_simple_iteration_method_mpi, check_validation_invalid_outputs_count) {
  boost::mpi::communicator world;
  size_t size = 3;
  // Create data
  std::vector<double> A = {4.0, 1.0, 2.0, 5.0, 9.0, 3.0, 4.0, 2.0, 12.0};
  std::vector<double> b = {1.0, 2.0, 1.0};
  std::vector<double> out(size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(b.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  }
  // Create Task
  rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), false);
}

TEST(rezantseva_a_simple_iteration_method_mpi, check_validation_invalid_matrix_size) {
  boost::mpi::communicator world;
  size_t size = 0;
  // Create data
  std::vector<double> A = {4.0, 1.0, 2.0, 5.0, 9.0, 3.0, 4.0, 2.0, 12.0};
  std::vector<double> b = {1.0, 2.0, 1.0};
  std::vector<double> out(size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(b.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  // Create Task
  rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), false);
}

TEST(rezantseva_a_simple_iteration_method_mpi, check_validation_invalid_matrix) {
  boost::mpi::communicator world;
  size_t size = 3;
  // Create data
  std::vector<double> A = {4.0, 1.0, 7.0, 5.0, 7.0, 3.0, 4.0, 2.0, 5.0};
  std::vector<double> b = {1.0, 2.0, 1.0};
  std::vector<double> out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(b.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  // Create Task
  rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), false);
}

TEST(rezantseva_a_simple_iteration_method_mpi, check_validation_zero_matrix) {
  boost::mpi::communicator world;
  size_t size = 3;
  // Create data
  std::vector<double> A = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> b = {1.0, 2.0, 1.0};
  std::vector<double> out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(b.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  // Create Task
  rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), false);
}

TEST(rezantseva_a_simple_iteration_method_mpi, check_validation_zero_diagonal_matrix) {
  boost::mpi::communicator world;
  size_t size = 3;
  // Create data
  std::vector<double> A = {0.0, 2.0, -2.0, 5.0, 0.0, 6.0, 7.0, -1.0, 0.0};
  std::vector<double> b = {1.0, 2.0, 1.0};
  std::vector<double> out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(b.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  // Create Task
  rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), false);
}