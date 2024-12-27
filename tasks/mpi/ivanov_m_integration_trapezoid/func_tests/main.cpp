// Copyright 2024 Ivanov Mike
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <numbers>
#include <vector>

#include "mpi/ivanov_m_integration_trapezoid/include/ops_mpi.hpp"

TEST(ivanov_m_integration_trapezoid_mpi_func_test, simple_parabola) {
  boost::mpi::communicator world;
  double a = 0;
  double b = 1;
  int n = 10000;

  // Create function y = x^2
  std::function<double(double)> _f = [](double x) { return x * x; };

  std::vector<double> global_vec = {a, b, static_cast<double>(n)};
  std::vector<double> global_result(1, 0.0);
  std::vector<double> reference_result(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  ivanov_m_integration_trapezoid_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.add_function(_f);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    // Create Task
    ivanov_m_integration_trapezoid_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    testMpiTaskSequential.add_function(_f);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  ASSERT_NEAR(reference_result[0], global_result[0], 1e-3);
}

TEST(ivanov_m_integration_trapezoid_mpi_func_test, simple_parabola_swapped_borders) {
  boost::mpi::communicator world;
  double a = 1;
  double b = 0;
  int n = 10000;

  // Create function y = x^2
  std::function<double(double)> _f = [](double x) { return x * x; };

  std::vector<double> global_vec = {a, b, static_cast<double>(n)};
  std::vector<double> global_result(1, 0.0);
  std::vector<double> reference_result(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  ivanov_m_integration_trapezoid_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.add_function(_f);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    // Create Task
    ivanov_m_integration_trapezoid_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    testMpiTaskSequential.add_function(_f);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  ASSERT_NEAR(reference_result[0], global_result[0], 1e-3);
}

TEST(ivanov_m_integration_trapezoid_mpi_func_test, line_function) {
  boost::mpi::communicator world;
  double a = 0;
  double b = 5;
  int n = 10000;

  // Create function y = 2*(x-1)
  std::function<double(double)> _f = [](double x) { return 2 * (x - 1); };

  std::vector<double> global_vec = {a, b, static_cast<double>(n)};
  std::vector<double> global_result(1, 0.0);
  std::vector<double> reference_result(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  ivanov_m_integration_trapezoid_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.add_function(_f);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    // Create Task
    ivanov_m_integration_trapezoid_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    testMpiTaskSequential.add_function(_f);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  ASSERT_NEAR(reference_result[0], global_result[0], 1e-3);
}

TEST(ivanov_m_integration_trapezoid_mpi_func_test, sinus) {
  boost::mpi::communicator world;
  double a = 0;
  auto b = static_cast<double>(std::numbers::pi);
  int n = 10000;

  // Create function y = sin(x)
  std::function<double(double)> _f = [](double x) { return sin(x); };

  std::vector<double> global_vec = {a, b, static_cast<double>(n)};
  std::vector<double> global_result(1, 0.0);
  std::vector<double> reference_result(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  ivanov_m_integration_trapezoid_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.add_function(_f);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    // Create Task
    ivanov_m_integration_trapezoid_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    testMpiTaskSequential.add_function(_f);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  ASSERT_NEAR(reference_result[0], global_result[0], 1e-3);
}

TEST(ivanov_m_integration_trapezoid_mpi_func_test, sqrt) {
  boost::mpi::communicator world;
  double a = 0;
  double b = 4;
  int n = 10000;

  // Create function y = sqrt(x)
  std::function<double(double)> _f = [](double x) { return sqrt(x); };

  std::vector<double> global_vec = {a, b, static_cast<double>(n)};
  std::vector<double> global_result(1, 0.0);
  std::vector<double> reference_result(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  ivanov_m_integration_trapezoid_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.add_function(_f);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    // Create Task
    ivanov_m_integration_trapezoid_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    testMpiTaskSequential.add_function(_f);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  ASSERT_NEAR(reference_result[0], global_result[0], 1e-3);
}

TEST(ivanov_m_integration_trapezoid_mpi_func_test, simple_ln) {
  boost::mpi::communicator world;
  double a = 1;
  auto b = static_cast<double>(std::numbers::e);
  int n = 10000;

  // Create function y = ln(x)
  std::function<double(double)> _f = [](double x) { return log(x); };

  std::vector<double> global_vec = {a, b, static_cast<double>(n)};
  std::vector<double> global_result(1, 0.0);
  std::vector<double> reference_result(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  ivanov_m_integration_trapezoid_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.add_function(_f);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    // Create Task
    ivanov_m_integration_trapezoid_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    testMpiTaskSequential.add_function(_f);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  ASSERT_NEAR(reference_result[0], global_result[0], 1e-3);
}

TEST(ivanov_m_integration_trapezoid_mpi_func_test, simple_ln_with_right_border_sqr_exp) {
  boost::mpi::communicator world;
  double a = 1;
  auto b = static_cast<double>(std::numbers::e * std::numbers::e);
  int n = 10000;

  // Create function y = ln(x)
  std::function<double(double)> _f = [](double x) { return log(x); };

  std::vector<double> global_vec = {a, b, static_cast<double>(n)};
  std::vector<double> global_result(1, 0.0);
  std::vector<double> reference_result(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  ivanov_m_integration_trapezoid_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.add_function(_f);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    // Create Task
    ivanov_m_integration_trapezoid_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    testMpiTaskSequential.add_function(_f);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  ASSERT_NEAR(reference_result[0], global_result[0], 1e-3);
}

TEST(ivanov_m_integration_trapezoid_mpi_func_test, equal_borders) {
  boost::mpi::communicator world;
  double a = 1;
  double b = 1;
  int n = 10000;

  // Create function y = x^2
  std::function<double(double)> _f = [](double x) { return x * x; };

  std::vector<double> global_vec = {a, b, static_cast<double>(n)};
  std::vector<double> global_result(1, 0.0);
  std::vector<double> reference_result(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  ivanov_m_integration_trapezoid_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.add_function(_f);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    // Create Task
    ivanov_m_integration_trapezoid_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    testMpiTaskSequential.add_function(_f);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  ASSERT_NEAR(reference_result[0], global_result[0], 1e-3);
}

TEST(ivanov_m_integration_trapezoid_mpi_func_test, parabola_with_large_result) {
  boost::mpi::communicator world;
  double a = 0;
  double b = 100;
  int n = 100000;

  // Create function y = x^2
  std::function<double(double)> _f = [](double x) { return x * x; };

  std::vector<double> global_vec = {a, b, static_cast<double>(n)};
  std::vector<double> global_result(1, 0.0);
  std::vector<double> reference_result(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  ivanov_m_integration_trapezoid_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.add_function(_f);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    // Create Task
    ivanov_m_integration_trapezoid_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    testMpiTaskSequential.add_function(_f);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  ASSERT_NEAR(reference_result[0], global_result[0], 1e-3);
}

TEST(ivanov_m_integration_trapezoid_mpi_func_test, cosinus_result_equals_zero) {
  boost::mpi::communicator world;
  double a = 0;
  auto b = static_cast<double>(std::numbers::pi);
  int n = 10000;

  // Create function y = cos(x)
  std::function<double(double)> _f = [](double x) { return cos(x); };

  std::vector<double> global_vec = {a, b, static_cast<double>(n)};
  std::vector<double> global_result(1, 0.0);
  std::vector<double> reference_result(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  ivanov_m_integration_trapezoid_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.add_function(_f);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    // Create Task
    ivanov_m_integration_trapezoid_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    testMpiTaskSequential.add_function(_f);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  ASSERT_NEAR(reference_result[0], global_result[0], 1e-3);
}

TEST(ivanov_m_integration_trapezoid_mpi_func_test, cosinus_result_less_than_zero) {
  boost::mpi::communicator world;
  double a = 0;
  auto b = static_cast<double>(std::numbers::pi);
  int n = 10000;

  // Create function y = cos(x) - 1
  std::function<double(double)> _f = [](double x) { return cos(x) - 1; };

  std::vector<double> global_vec = {a, b, static_cast<double>(n)};
  std::vector<double> global_result(1, 0.0);
  std::vector<double> reference_result(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  ivanov_m_integration_trapezoid_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.add_function(_f);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    // Create Task
    ivanov_m_integration_trapezoid_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    testMpiTaskSequential.add_function(_f);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  ASSERT_NEAR(reference_result[0], global_result[0], 1e-3);
}