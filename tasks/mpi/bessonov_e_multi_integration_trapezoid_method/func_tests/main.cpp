#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include "mpi/bessonov_e_multi_integration_trapezoid_method/include/ops_mpi.hpp"

TEST(bessonov_e_multi_integration_trapezoid_method_mpi, TwoDimensional) {
  boost::mpi::communicator world;
  size_t dim = 2;
  std::vector<double> lower_limits = {0.0, 0.0};
  std::vector<double> upper_limits = {1.0, 1.0};
  std::vector<int> steps = {50, 50};
  double parallel_result = 0.0;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
    taskDataPar->inputs_count.emplace_back(lower_limits.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
    taskDataPar->inputs_count.emplace_back(upper_limits.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    taskDataPar->inputs_count.emplace_back(steps.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&parallel_result));
  }

  bessonov_e_multi_integration_trapezoid_method_mpi::TestMPITaskParallel parallelTask(taskDataPar);
  parallelTask.integrand = [](const std::vector<double>& point) { return point[0] + point[1]; };

  ASSERT_TRUE(parallelTask.validation());
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    double sequential_result = 0.0;
    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
    taskDataSeq->inputs_count.emplace_back(lower_limits.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
    taskDataSeq->inputs_count.emplace_back(upper_limits.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    taskDataSeq->inputs_count.emplace_back(steps.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&sequential_result));

    bessonov_e_multi_integration_trapezoid_method_mpi::TestMPITaskSequential sequentialTask(taskDataSeq);
    sequentialTask.integrand = [](const std::vector<double>& point) { return point[0] + point[1]; };

    ASSERT_TRUE(sequentialTask.validation());
    ASSERT_TRUE(sequentialTask.pre_processing());
    ASSERT_TRUE(sequentialTask.run());
    ASSERT_TRUE(sequentialTask.post_processing());

    ASSERT_NEAR(parallel_result, sequential_result, 1e-2);
  }
}

TEST(bessonov_e_multi_integration_trapezoid_method_mpi, ThreeDimensional) {
  boost::mpi::communicator world;
  size_t dim = 3;
  std::vector<double> lower_limits = {0.0, 0.0, 0.0};
  std::vector<double> upper_limits = {1.0, 1.0, 1.0};
  std::vector<int> steps = {50, 50, 50};
  double parallel_result = 0.0;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
    taskDataPar->inputs_count.emplace_back(lower_limits.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
    taskDataPar->inputs_count.emplace_back(upper_limits.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    taskDataPar->inputs_count.emplace_back(steps.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&parallel_result));
  }

  bessonov_e_multi_integration_trapezoid_method_mpi::TestMPITaskParallel parallelTask(taskDataPar);
  parallelTask.integrand = [](const std::vector<double>& point) { return point[0] + point[1] + point[2]; };

  ASSERT_TRUE(parallelTask.validation());
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    double sequential_result = 0.0;
    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
    taskDataSeq->inputs_count.emplace_back(lower_limits.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
    taskDataSeq->inputs_count.emplace_back(upper_limits.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    taskDataSeq->inputs_count.emplace_back(steps.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&sequential_result));

    bessonov_e_multi_integration_trapezoid_method_mpi::TestMPITaskSequential sequentialTask(taskDataSeq);
    sequentialTask.integrand = [](const std::vector<double>& point) { return point[0] + point[1] + point[2]; };

    ASSERT_TRUE(sequentialTask.validation());
    ASSERT_TRUE(sequentialTask.pre_processing());
    ASSERT_TRUE(sequentialTask.run());
    ASSERT_TRUE(sequentialTask.post_processing());

    ASSERT_NEAR(parallel_result, sequential_result, 1e-2);
  }
}

TEST(bessonov_e_multi_integration_trapezoid_method_mpi, TwoDimensionalComplexFunction) {
  boost::mpi::communicator world;
  size_t dim = 2;
  std::vector<double> lower_limits = {-1.0, -1.0};
  std::vector<double> upper_limits = {1.0, 1.0};
  std::vector<int> steps = {100, 100};
  double parallel_result = 0.0;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
    taskDataPar->inputs_count.emplace_back(lower_limits.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
    taskDataPar->inputs_count.emplace_back(upper_limits.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    taskDataPar->inputs_count.emplace_back(steps.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&parallel_result));
  }

  bessonov_e_multi_integration_trapezoid_method_mpi::TestMPITaskParallel parallelTask(taskDataPar);
  parallelTask.integrand = [](const std::vector<double>& point) {
    return std::exp(-point[0] * point[0] - point[1] * point[1]);
  };

  ASSERT_TRUE(parallelTask.validation());
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    double sequential_result = 0.0;
    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
    taskDataSeq->inputs_count.emplace_back(lower_limits.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
    taskDataSeq->inputs_count.emplace_back(upper_limits.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    taskDataSeq->inputs_count.emplace_back(steps.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&sequential_result));

    bessonov_e_multi_integration_trapezoid_method_mpi::TestMPITaskSequential sequentialTask(taskDataSeq);
    sequentialTask.integrand = [](const std::vector<double>& point) {
      return std::exp(-point[0] * point[0] - point[1] * point[1]);
    };

    ASSERT_TRUE(sequentialTask.validation());
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    ASSERT_NEAR(parallel_result, sequential_result, 1e-2);
  }
}

TEST(bessonov_e_multi_integration_trapezoid_method_mpi, ThreeDimensionalNonLinearFunction) {
  boost::mpi::communicator world;
  size_t dim = 3;
  std::vector<double> lower_limits = {0.0, 0.0, 0.0};
  std::vector<double> upper_limits = {2.0, 2.0, 2.0};
  std::vector<int> steps = {50, 50, 50};
  double parallel_result = 0.0;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
    taskDataPar->inputs_count.emplace_back(lower_limits.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
    taskDataPar->inputs_count.emplace_back(upper_limits.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    taskDataPar->inputs_count.emplace_back(steps.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&parallel_result));
  }

  bessonov_e_multi_integration_trapezoid_method_mpi::TestMPITaskParallel parallelTask(taskDataPar);
  parallelTask.integrand = [](const std::vector<double>& point) {
    return std::sin(point[0]) * std::cos(point[1]) * std::exp(-point[2]);
  };

  ASSERT_TRUE(parallelTask.validation());
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    double sequential_result = 0.0;
    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
    taskDataSeq->inputs_count.emplace_back(lower_limits.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
    taskDataSeq->inputs_count.emplace_back(upper_limits.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    taskDataSeq->inputs_count.emplace_back(steps.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&sequential_result));

    bessonov_e_multi_integration_trapezoid_method_mpi::TestMPITaskSequential sequentialTask(taskDataSeq);
    sequentialTask.integrand = [](const std::vector<double>& point) {
      return std::sin(point[0]) * std::cos(point[1]) * std::exp(-point[2]);
    };

    ASSERT_TRUE(sequentialTask.validation());
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    ASSERT_NEAR(parallel_result, sequential_result, 1e-2);
  }
}

TEST(bessonov_e_multi_integration_trapezoid_method_mpi, FourDimensionalComplexFunction) {
  boost::mpi::communicator world;
  size_t dim = 4;
  std::vector<double> lower_limits = {-1.0, 0.0, -1.0, 0.0};
  std::vector<double> upper_limits = {1.0, 1.0, 1.0, 2.0};
  std::vector<int> steps = {50, 50, 50, 50};
  double parallel_result = 0.0;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
    taskDataPar->inputs_count.emplace_back(lower_limits.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
    taskDataPar->inputs_count.emplace_back(upper_limits.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    taskDataPar->inputs_count.emplace_back(steps.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&parallel_result));
  }

  bessonov_e_multi_integration_trapezoid_method_mpi::TestMPITaskParallel parallelTask(taskDataPar);
  parallelTask.integrand = [](const std::vector<double>& point) {
    return std::sin(point[0]) * point[1] + std::cos(point[2]) * std::exp(-point[3]);
  };

  ASSERT_TRUE(parallelTask.validation());
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    double sequential_result = 0.0;
    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
    taskDataSeq->inputs_count.emplace_back(lower_limits.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
    taskDataSeq->inputs_count.emplace_back(upper_limits.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    taskDataSeq->inputs_count.emplace_back(steps.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&sequential_result));

    bessonov_e_multi_integration_trapezoid_method_mpi::TestMPITaskSequential sequentialTask(taskDataSeq);
    sequentialTask.integrand = [](const std::vector<double>& point) {
      return std::sin(point[0]) * point[1] + std::cos(point[2]) * std::exp(-point[3]);
    };

    ASSERT_TRUE(sequentialTask.validation());
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    ASSERT_NEAR(parallel_result, sequential_result, 1e-2);
  }
}

TEST(bessonov_e_multi_integration_trapezoid_method_mpi, FiveDimensionalComplexFunction) {
  boost::mpi::communicator world;
  size_t dim = 5;
  std::vector<double> lower_limits = {0.0, -2.0, -1.0, 0.0, 1.0};
  std::vector<double> upper_limits = {1.0, 2.0, 1.0, 3.0, 4.0};
  std::vector<int> steps = {20, 20, 20, 20, 20};
  double parallel_result = 0.0;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
    taskDataPar->inputs_count.emplace_back(lower_limits.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
    taskDataPar->inputs_count.emplace_back(upper_limits.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    taskDataPar->inputs_count.emplace_back(steps.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&parallel_result));
  }

  bessonov_e_multi_integration_trapezoid_method_mpi::TestMPITaskParallel parallelTask(taskDataPar);
  parallelTask.integrand = [](const std::vector<double>& point) {
    return std::sin(point[0] * point[1]) + std::cos(point[2] * point[3]) * std::exp(-point[4]);
  };

  ASSERT_TRUE(parallelTask.validation());
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    double sequential_result = 0.0;
    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
    taskDataSeq->inputs_count.emplace_back(lower_limits.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
    taskDataSeq->inputs_count.emplace_back(upper_limits.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    taskDataSeq->inputs_count.emplace_back(steps.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&sequential_result));

    bessonov_e_multi_integration_trapezoid_method_mpi::TestMPITaskSequential sequentialTask(taskDataSeq);
    sequentialTask.integrand = [](const std::vector<double>& point) {
      return std::sin(point[0] * point[1]) + std::cos(point[2] * point[3]) * std::exp(-point[4]);
    };

    ASSERT_TRUE(sequentialTask.validation());
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    ASSERT_NEAR(parallel_result, sequential_result, 1e-2);
  }
}

TEST(bessonov_e_multi_integration_trapezoid_method_mpi, SixDimensionalComplexFunction) {
  boost::mpi::communicator world;
  size_t dim = 6;
  std::vector<double> lower_limits = {-2.0, 0.0, -3.0, 1.0, -1.0, 0.0};
  std::vector<double> upper_limits = {2.0, 1.0, 3.0, 4.0, 2.0, 1.0};
  std::vector<int> steps = {10, 10, 10, 10, 10, 10};
  double parallel_result = 0.0;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
    taskDataPar->inputs_count.emplace_back(lower_limits.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
    taskDataPar->inputs_count.emplace_back(upper_limits.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    taskDataPar->inputs_count.emplace_back(steps.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&parallel_result));
  }

  bessonov_e_multi_integration_trapezoid_method_mpi::TestMPITaskParallel parallelTask(taskDataPar);
  parallelTask.integrand = [](const std::vector<double>& point) {
    return point[0] * point[1] - point[2] * point[3] + std::sin(point[4] * point[5]);
  };

  ASSERT_TRUE(parallelTask.validation());
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    double sequential_result = 0.0;
    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
    taskDataSeq->inputs_count.emplace_back(lower_limits.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
    taskDataSeq->inputs_count.emplace_back(upper_limits.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    taskDataSeq->inputs_count.emplace_back(steps.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&sequential_result));

    bessonov_e_multi_integration_trapezoid_method_mpi::TestMPITaskSequential sequentialTask(taskDataSeq);
    sequentialTask.integrand = [](const std::vector<double>& point) {
      return point[0] * point[1] - point[2] * point[3] + std::sin(point[4] * point[5]);
    };

    ASSERT_TRUE(sequentialTask.validation());
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    ASSERT_NEAR(parallel_result, sequential_result, 1e-2);
  }
}

TEST(bessonov_e_multi_integration_trapezoid_method_mpi, RandomIntervalThreeDimensionalWithEqualCheck) {
  boost::mpi::communicator world;
  size_t dim = 3;
  std::vector<double> lower_limits(dim);
  std::vector<double> upper_limits(dim);
  std::vector<int> steps = {50, 50, 50};
  double parallel_result = 0.0;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 10.0);

  for (size_t i = 0; i < dim; ++i) {
    lower_limits[i] = dis(gen);
    upper_limits[i] = dis(gen);

    if (std::abs(lower_limits[i] - upper_limits[i]) < 1e-6) {
      upper_limits[i] += 1.0;
    }

    if (lower_limits[i] > upper_limits[i]) {
      std::swap(lower_limits[i], upper_limits[i]);
    }
  }

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
    taskDataPar->inputs_count.emplace_back(lower_limits.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
    taskDataPar->inputs_count.emplace_back(upper_limits.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    taskDataPar->inputs_count.emplace_back(steps.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&parallel_result));
  }

  bessonov_e_multi_integration_trapezoid_method_mpi::TestMPITaskParallel parallelTask(taskDataPar);
  parallelTask.integrand = [](const std::vector<double>& point) { return point[0] + point[1] + point[2]; };

  ASSERT_TRUE(parallelTask.validation());
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    double sequential_result = 0.0;
    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
    taskDataSeq->inputs_count.emplace_back(lower_limits.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
    taskDataSeq->inputs_count.emplace_back(upper_limits.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    taskDataSeq->inputs_count.emplace_back(steps.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&sequential_result));

    bessonov_e_multi_integration_trapezoid_method_mpi::TestMPITaskSequential sequentialTask(taskDataSeq);
    sequentialTask.integrand = [](const std::vector<double>& point) { return point[0] + point[1] + point[2]; };

    ASSERT_TRUE(sequentialTask.validation());
    ASSERT_TRUE(sequentialTask.pre_processing());
    ASSERT_TRUE(sequentialTask.run());
    ASSERT_TRUE(sequentialTask.post_processing());

    ASSERT_NEAR(parallel_result, sequential_result, 1e-2);
  }
}

TEST(bessonov_e_multi_integration_trapezoid_method_mpi, ValidationTestInvalidBounds) {
  boost::mpi::communicator world;
  size_t dim = 2;
  std::vector<double> lower_limits = {1.0, 0.0};
  std::vector<double> upper_limits = {0.0, 1.0};
  std::vector<int> steps = {50, 50};
  double result = 0.0;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
    taskData->inputs_count.emplace_back(lower_limits.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
    taskData->inputs_count.emplace_back(upper_limits.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    taskData->inputs_count.emplace_back(steps.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  }

  bessonov_e_multi_integration_trapezoid_method_mpi::TestMPITaskParallel task(taskData);

  if (world.rank() == 0) {
    ASSERT_FALSE(task.validation());
  }
}

TEST(bessonov_e_multi_integration_trapezoid_method_mpi, ValidationTestValidData) {
  boost::mpi::communicator world;
  size_t dim = 2;
  std::vector<double> lower_limits = {0.0, 0.0};
  std::vector<double> upper_limits = {1.0, 1.0};
  std::vector<int> steps = {50, 50};
  double result = 0.0;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
    taskData->inputs_count.emplace_back(lower_limits.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
    taskData->inputs_count.emplace_back(upper_limits.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    taskData->inputs_count.emplace_back(steps.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  }

  bessonov_e_multi_integration_trapezoid_method_mpi::TestMPITaskParallel task(taskData);

  if (world.rank() == 0) {
    ASSERT_TRUE(task.validation());
  }
}

TEST(bessonov_e_multi_integration_trapezoid_method_mpi, ValidationTestInvalidSteps) {
  boost::mpi::communicator world;
  size_t dim = 2;
  std::vector<double> lower_limits = {0.0, 0.0};
  std::vector<double> upper_limits = {1.0, 1.0};
  std::vector<int> steps = {10, -1};
  double result = 0.0;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
    taskData->inputs_count.emplace_back(lower_limits.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
    taskData->inputs_count.emplace_back(upper_limits.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    taskData->inputs_count.emplace_back(steps.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  }

  bessonov_e_multi_integration_trapezoid_method_mpi::TestMPITaskParallel task(taskData);

  if (world.rank() == 0) {
    ASSERT_FALSE(task.validation());
  }
}

TEST(bessonov_e_multi_integration_trapezoid_method_mpi, ValidationTestInvalidCount) {
  boost::mpi::communicator world;
  size_t dim = 2;
  std::vector<double> lower_limits = {0.0, 0.0};
  std::vector<double> upper_limits = {1.0};
  std::vector<int> steps = {10, 10};
  double result = 0.0;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
    taskData->inputs_count.emplace_back(lower_limits.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
    taskData->inputs_count.emplace_back(upper_limits.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    taskData->inputs_count.emplace_back(steps.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  }

  bessonov_e_multi_integration_trapezoid_method_mpi::TestMPITaskParallel task(taskData);

  if (world.rank() == 0) {
    ASSERT_FALSE(task.validation());
  }
}

TEST(bessonov_e_multi_integration_trapezoid_method_mpi, ValidationTestExtraOutputs) {
  boost::mpi::communicator world;
  size_t dim = 2;
  std::vector<double> lower_limits = {0.0, 0.0};
  std::vector<double> upper_limits = {1.0, 1.0};
  std::vector<int> steps = {10, 10};
  double result1 = 0.0;
  double result2 = 0.0;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
    taskData->inputs_count.emplace_back(lower_limits.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
    taskData->inputs_count.emplace_back(upper_limits.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    taskData->inputs_count.emplace_back(steps.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result1));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result2));
  }

  bessonov_e_multi_integration_trapezoid_method_mpi::TestMPITaskParallel task(taskData);

  if (world.rank() == 0) {
    ASSERT_FALSE(task.validation());
  }
}