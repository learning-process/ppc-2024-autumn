#include <gtest/gtest.h>

#include "mpi/voroshilov_v_bivariate_optimization_by_area/include/ops_mpi.hpp"

bool validation_test_mpi(std::vector<char> q_vec, size_t g_count, std::vector<std::vector<char>> g_vec,
                         std::vector<double> areas_vec, std::vector<int> steps_vec) {
  boost::mpi::communicator comm;

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  // Criterium-function:
  taskDataParallel->inputs_count.emplace_back(q_vec.size());
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(q_vec.data()));
  // Count of constraints-functions:
  taskDataParallel->inputs_count.emplace_back(1);
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(&g_count));
  // Constraints-functions:
  for (size_t i = 0; i < g_vec.size(); i++) {
    taskDataParallel->inputs_count.emplace_back(g_vec[i].size());
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(g_vec[i].data()));
  }
  std::vector<double> optimum_vec(3);
  if (comm.rank() == 0) {
    // Search area boundaries:
    taskDataParallel->inputs_count.emplace_back(areas_vec.size());
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(areas_vec.data()));
    // Steps counts (how many points will be used):
    taskDataParallel->inputs_count.emplace_back(steps_vec.size());
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(steps_vec.data()));
    // Output - optimum point and value:
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(optimum_vec.data()));
  }
  voroshilov_v_bivariate_optimization_by_area_mpi::OptimizationMPITaskParallel optimizationMPITaskParallel(
      taskDataParallel);
  return optimizationMPITaskParallel.validation();
}

bool validation_test_seq(std::vector<char> q_vec, size_t g_count, std::vector<std::vector<char>> g_vec,
                         std::vector<double> areas_vec, std::vector<int> steps_vec) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  // Criterium-function:
  taskDataSeq->inputs_count.emplace_back(q_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(q_vec.data()));
  // Count of constraints-functions:
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&g_count));
  // Constraints-functions:
  for (size_t i = 0; i < g_vec.size(); i++) {
    taskDataSeq->inputs_count.emplace_back(g_vec[i].size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(g_vec[i].data()));
  }
  // Search area boundaries:
  taskDataSeq->inputs_count.emplace_back(areas_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(areas_vec.data()));
  // Steps counts (how many points will be used):
  taskDataSeq->inputs_count.emplace_back(steps_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(steps_vec.data()));
  // Output - optimum point and value:
  std::vector<double> optimum_vec(3);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(optimum_vec.data()));

  voroshilov_v_bivariate_optimization_by_area_mpi::OptimizationMPITaskSequential optimizationMPITaskSequential(
      taskDataSeq);
  return optimizationMPITaskSequential.validation();
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_validation_empty_criterium_function) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str;
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::string g_str1 = "-x^1y^0";
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1";
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  size_t g_count = g_vec.size();

  // Search areas:
  std::vector<double> areas_vec({-10.0, 10.0, -10.0, 10.0});
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({250, 250});

  ASSERT_FALSE(validation_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec));

  if (world.rank() == 0) {
    ASSERT_FALSE(validation_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec));
  }
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_validation_incorrect_num_of_areas) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str = "x^2y^0 +x^0y^2";
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::string g_str1 = "-x^1y^0";
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1";
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  size_t g_count = g_vec.size();

  // Search areas:
  std::vector<double> areas_vec({-10.0, 10.0, -10.0});
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({250, 250});

  ASSERT_FALSE(validation_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec));

  if (world.rank() == 0) {
    ASSERT_FALSE(validation_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec));
  }
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_validation_incorrect_areas) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str = "x^2y^0 +x^0y^2";
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::string g_str1 = "-x^1y^0";
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1";
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  size_t g_count = g_vec.size();

  // Search areas:
  std::vector<double> areas_vec({-10.0, -20.0, -10.0, 10.0});
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({250, 250});

  ASSERT_FALSE(validation_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec));

  if (world.rank() == 0) {
    ASSERT_FALSE(validation_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec));
  }
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_validation_incorrect_num_of_steps_counts) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str = "x^2y^0 +x^0y^2";
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::string g_str1 = "-x^1y^0";
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1";
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  size_t g_count = g_vec.size();

  // Search areas:
  std::vector<double> areas_vec({-10.0, 10.0, -10.0, 10.0});
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({250});

  ASSERT_FALSE(validation_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec));

  if (world.rank() == 0) {
    ASSERT_FALSE(validation_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec));
  }
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_validation_incorrect_steps_counts) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str = "x^2y^0 +x^0y^2";
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::string g_str1 = "-x^1y^0";
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1";
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  size_t g_count = g_vec.size();

  // Search areas:
  std::vector<double> areas_vec({-10.0, 10.0, -10.0, 10.0});
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({0, 250});

  ASSERT_FALSE(validation_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec));

  if (world.rank() == 0) {
    ASSERT_FALSE(validation_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec));
  }
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_validation_incorrect_g_count) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str = "x^2y^0 +x^0y^2";
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::string g_str1 = "-x^1y^0";
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1";
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  size_t g_count = 33;

  // Search areas:
  std::vector<double> areas_vec({-10.0, 10.0, -10.0, 10.0});
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({250, 250});

  ASSERT_FALSE(validation_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec));

  if (world.rank() == 0) {
    ASSERT_FALSE(validation_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec));
  }
}

std::vector<double> run_test_mpi(std::vector<char> q_vec, size_t g_count, std::vector<std::vector<char>> g_vec,
                                 std::vector<double> areas_vec, std::vector<int> steps_vec) {
  boost::mpi::communicator comm;

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  // Criterium-function:
  taskDataParallel->inputs_count.emplace_back(q_vec.size());
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(q_vec.data()));
  // Count of constraints-functions:
  taskDataParallel->inputs_count.emplace_back(1);
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(&g_count));
  // Constraints-functions:
  for (size_t i = 0; i < g_vec.size(); i++) {
    taskDataParallel->inputs_count.emplace_back(g_vec[i].size());
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(g_vec[i].data()));
  }
  std::vector<double> optimum_vec(3);
  if (comm.rank() == 0) {
    // Search area boundaries:
    taskDataParallel->inputs_count.emplace_back(areas_vec.size());
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(areas_vec.data()));
    // Steps counts (how many points will be used):
    taskDataParallel->inputs_count.emplace_back(steps_vec.size());
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(steps_vec.data()));
    // Output - optimum point and value:
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(optimum_vec.data()));
  }
  voroshilov_v_bivariate_optimization_by_area_mpi::OptimizationMPITaskParallel optimizationMPITaskParallel(
      taskDataParallel);
  optimizationMPITaskParallel.validation();
  optimizationMPITaskParallel.pre_processing();
  optimizationMPITaskParallel.run();
  optimizationMPITaskParallel.post_processing();

  boost::mpi::broadcast(comm, optimum_vec.data(), optimum_vec.size(), 0);

  return optimum_vec;
}

std::vector<double> run_test_seq(std::vector<char> q_vec, size_t g_count, std::vector<std::vector<char>> g_vec,
                                 std::vector<double> areas_vec, std::vector<int> steps_vec) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  // Criterium-function:
  taskDataSeq->inputs_count.emplace_back(q_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(q_vec.data()));
  // Count of constraints-functions:
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&g_count));
  // Constraints-functions:
  for (size_t i = 0; i < g_vec.size(); i++) {
    taskDataSeq->inputs_count.emplace_back(g_vec[i].size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(g_vec[i].data()));
  }
  // Search area boundaries:
  taskDataSeq->inputs_count.emplace_back(areas_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(areas_vec.data()));
  // Steps counts (how many points will be used):
  taskDataSeq->inputs_count.emplace_back(steps_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(steps_vec.data()));
  // Output - optimum point and value:
  std::vector<double> optimum_vec(3);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(optimum_vec.data()));

  voroshilov_v_bivariate_optimization_by_area_mpi::OptimizationMPITaskSequential optimizationMPITaskSequential(
      taskDataSeq);
  optimizationMPITaskSequential.validation();
  optimizationMPITaskSequential.pre_processing();
  optimizationMPITaskSequential.run();
  optimizationMPITaskSequential.post_processing();

  return optimum_vec;
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_task_run_zero_func_without_constraints) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str = "0";
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::vector<std::vector<char>> g_vec;
  size_t g_count = g_vec.size();

  // Search areas:
  std::vector<double> areas_vec({-10.0, 10.0, -10.0, 10.0});
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({250, 250});
  // Output value:
  std::vector<double> optimum_vec_mpi = run_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec);

  double optimum_value_mpi = optimum_vec_mpi[2];

  double eps = 0.2;

  ASSERT_NEAR(optimum_value_mpi, 0.0, eps);  // dont check optimum point because it is everywhere

  if (world.rank() == 0) {
    std::vector<double> optimum_vec_seq = run_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec);

    double optimum_value_seq = optimum_vec_seq[2];

    ASSERT_EQ(optimum_value_seq, optimum_value_mpi);
  }
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_task_run_zero_func_with_constraints) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str = "0";
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::string g_str1 = "-x^1y^0 +1";  // x >= 1
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1 +1";  // y >= 1
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  size_t g_count = g_vec.size();

  // Search areas:
  std::vector<double> areas_vec({-10.0, 10.0, -10.0, 10.0});
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({250, 250});
  // Output value:
  std::vector<double> optimum_vec_mpi = run_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec);

  double optimum_x_mpi = optimum_vec_mpi[0];
  double optimum_y_mpi = optimum_vec_mpi[1];
  double optimum_value_mpi = optimum_vec_mpi[2];

  double eps = 0.2;

  ASSERT_TRUE(optimum_x_mpi >= 1.0);
  ASSERT_TRUE(optimum_y_mpi >= 1.0);
  ASSERT_NEAR(optimum_value_mpi, 0.0, eps);

  if (world.rank() == 0) {
    std::vector<double> optimum_vec_seq = run_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec);

    double optimum_x_seq = optimum_vec_seq[0];
    double optimum_y_seq = optimum_vec_seq[1];
    double optimum_value_seq = optimum_vec_seq[2];

    ASSERT_EQ(optimum_x_seq, optimum_x_mpi);
    ASSERT_EQ(optimum_y_seq, optimum_y_mpi);
    ASSERT_EQ(optimum_value_seq, optimum_value_mpi);
  }
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_task_run_paraboloid_without_constraints) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str = "x^2y^0 +x^0y^2";  // paraboloid x^2+y^2, increases from point (0;0)
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::vector<std::vector<char>> g_vec;
  size_t g_count = g_vec.size();

  // Search areas:
  std::vector<double> areas_vec({-5.0, 5.0, -5.0, 5.0});
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({250, 250});
  // Output value:
  std::vector<double> optimum_vec_mpi = run_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec);

  double optimum_x_mpi = optimum_vec_mpi[0];
  double optimum_y_mpi = optimum_vec_mpi[1];
  double optimum_value_mpi = optimum_vec_mpi[2];

  double eps = 0.2;

  ASSERT_NEAR(optimum_x_mpi, 0.0, eps);
  ASSERT_NEAR(optimum_y_mpi, 0.0, eps);
  ASSERT_NEAR(optimum_value_mpi, 0.0, eps);

  if (world.rank() == 0) {
    std::vector<double> optimum_vec_seq = run_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec);

    double optimum_x_seq = optimum_vec_seq[0];
    double optimum_y_seq = optimum_vec_seq[1];
    double optimum_value_seq = optimum_vec_seq[2];

    ASSERT_EQ(optimum_x_seq, optimum_x_mpi);
    ASSERT_EQ(optimum_y_seq, optimum_y_mpi);
    ASSERT_EQ(optimum_value_seq, optimum_value_mpi);
  }
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_task_run_paraboloid_with_constraints) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str = "x^2y^0 +x^0y^2";  // paraboloid x^2+y^2, increases from point (0;0)
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::string g_str1 = "-x^1y^0 +1";  // x >= 1
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1 +1";  // y >= 1
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  size_t g_count = g_vec.size();

  // Search areas:
  std::vector<double> areas_vec({-5.0, 5.0, -5.0, 5.0});
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({250, 250});
  // Output value:
  std::vector<double> optimum_vec_mpi = run_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec);

  double optimum_x_mpi = optimum_vec_mpi[0];
  double optimum_y_mpi = optimum_vec_mpi[1];
  double optimum_value_mpi = optimum_vec_mpi[2];

  double eps = 0.2;

  ASSERT_NEAR(optimum_x_mpi, 1.0, eps);
  ASSERT_NEAR(optimum_y_mpi, 1.0, eps);
  ASSERT_NEAR(optimum_value_mpi, 2.0, eps);

  if (world.rank() == 0) {
    std::vector<double> optimum_vec_seq = run_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec);

    double optimum_x_seq = optimum_vec_seq[0];
    double optimum_y_seq = optimum_vec_seq[1];
    double optimum_value_seq = optimum_vec_seq[2];

    ASSERT_EQ(optimum_x_seq, optimum_x_mpi);
    ASSERT_EQ(optimum_y_seq, optimum_y_mpi);
    ASSERT_EQ(optimum_value_seq, optimum_value_mpi);
  }
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_task_run_paraboloid_degree_of_2) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str = "x^2y^0 +x^0y^2";  // paraboloid x^2+y^2, increases from point (0;0)
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::string g_str1 = "-x^1y^0 +1";  // x >= 1
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1 +1";  // y >= 1
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  size_t g_count = g_vec.size();

  // Search areas:
  std::vector<double> areas_vec({-5.0, 5.0, -5.0, 5.0});
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({256, 256});
  // Output value:
  std::vector<double> optimum_vec_mpi = run_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec);

  double optimum_x_mpi = optimum_vec_mpi[0];
  double optimum_y_mpi = optimum_vec_mpi[1];
  double optimum_value_mpi = optimum_vec_mpi[2];

  double eps = 0.2;

  ASSERT_NEAR(optimum_x_mpi, 1.0, eps);
  ASSERT_NEAR(optimum_y_mpi, 1.0, eps);
  ASSERT_NEAR(optimum_value_mpi, 2.0, eps);

  if (world.rank() == 0) {
    std::vector<double> optimum_vec_seq = run_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec);

    double optimum_x_seq = optimum_vec_seq[0];
    double optimum_y_seq = optimum_vec_seq[1];
    double optimum_value_seq = optimum_vec_seq[2];

    ASSERT_EQ(optimum_x_seq, optimum_x_mpi);
    ASSERT_EQ(optimum_y_seq, optimum_y_mpi);
    ASSERT_EQ(optimum_value_seq, optimum_value_mpi);
  }
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_task_run_paraboloid_minus_number_without_constraints) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str = "x^2y^0 +x^0y^2 -10";  // paraboloid x^2+y^2 minus 10, increases from -10
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::vector<std::vector<char>> g_vec;
  size_t g_count = g_vec.size();

  // Search areas:
  std::vector<double> areas_vec({-5.0, 5.0, -5.0, 5.0});
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({250, 250});
  // Output value:
  std::vector<double> optimum_vec_mpi = run_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec);

  double optimum_x_mpi = optimum_vec_mpi[0];
  double optimum_y_mpi = optimum_vec_mpi[1];
  double optimum_value_mpi = optimum_vec_mpi[2];

  double eps = 0.2;

  ASSERT_NEAR(optimum_x_mpi, 0.0, eps);
  ASSERT_NEAR(optimum_y_mpi, 0.0, eps);
  ASSERT_NEAR(optimum_value_mpi, -10.0, eps);

  if (world.rank() == 0) {
    std::vector<double> optimum_vec_seq = run_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec);

    double optimum_x_seq = optimum_vec_seq[0];
    double optimum_y_seq = optimum_vec_seq[1];
    double optimum_value_seq = optimum_vec_seq[2];

    ASSERT_EQ(optimum_x_seq, optimum_x_mpi);
    ASSERT_EQ(optimum_y_seq, optimum_y_mpi);
    ASSERT_EQ(optimum_value_seq, optimum_value_mpi);
  }
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_task_run_paraboloid_minus_number_with_constraints) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str = "x^2y^0 +x^0y^2 -10";  // paraboloid x^2+y^2 munus 10 (increases from -10)
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::string g_str1 = "-x^1y^0 +1";  // x >= 1
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1 +1";  // y >= 1
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  size_t g_count = g_vec.size();

  // Search areas:
  std::vector<double> areas_vec({-5.0, 5.0, -5.0, 5.0});
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({250, 250});
  // Output value:
  std::vector<double> optimum_vec_mpi = run_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec);

  double optimum_x_mpi = optimum_vec_mpi[0];
  double optimum_y_mpi = optimum_vec_mpi[1];
  double optimum_value_mpi = optimum_vec_mpi[2];

  double eps = 0.2;

  ASSERT_NEAR(optimum_x_mpi, 1.0, eps);
  ASSERT_NEAR(optimum_y_mpi, 1.0, eps);
  ASSERT_NEAR(optimum_value_mpi, -8.0, eps);

  if (world.rank() == 0) {
    std::vector<double> optimum_vec_seq = run_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec);

    double optimum_x_seq = optimum_vec_seq[0];
    double optimum_y_seq = optimum_vec_seq[1];
    double optimum_value_seq = optimum_vec_seq[2];

    ASSERT_EQ(optimum_x_seq, optimum_x_mpi);
    ASSERT_EQ(optimum_y_seq, optimum_y_mpi);
    ASSERT_EQ(optimum_value_seq, optimum_value_mpi);
  }
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_task_run_paraboloid_minus_number_different_steps) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str = "x^2y^0 +x^0y^2 -10";  // paraboloid x^2+y^2 munus 10 (increases from -10)
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::string g_str1 = "-x^1y^0 +1";  // x >= 1
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1 +1";  // y >= 1
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  size_t g_count = g_vec.size();

  // Search areas:
  std::vector<double> areas_vec({-5.0, 5.0, -5.0, 5.0});
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({240, 260});
  // Output value:
  std::vector<double> optimum_vec_mpi = run_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec);

  double optimum_x_mpi = optimum_vec_mpi[0];
  double optimum_y_mpi = optimum_vec_mpi[1];
  double optimum_value_mpi = optimum_vec_mpi[2];

  double eps = 0.2;

  ASSERT_NEAR(optimum_x_mpi, 1.0, eps);
  ASSERT_NEAR(optimum_y_mpi, 1.0, eps);
  ASSERT_NEAR(optimum_value_mpi, -8.0, eps);

  if (world.rank() == 0) {
    std::vector<double> optimum_vec_seq = run_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec);

    double optimum_x_seq = optimum_vec_seq[0];
    double optimum_y_seq = optimum_vec_seq[1];
    double optimum_value_seq = optimum_vec_seq[2];

    ASSERT_EQ(optimum_x_seq, optimum_x_mpi);
    ASSERT_EQ(optimum_y_seq, optimum_y_mpi);
    ASSERT_EQ(optimum_value_seq, optimum_value_mpi);
  }
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_task_run_shifted_paraboloid_without_constraints) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str = "x^2y^0 -12x^1y^0 +x^0y^2 -4x^0y^1";  // shifted paraboloid, increases from about -40
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::vector<std::vector<char>> g_vec;
  size_t g_count = g_vec.size();

  // Search areas:
  std::vector<double> areas_vec({-10.0, 10.0, -10.0, 10.0});
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({250, 250});
  // Output value:
  std::vector<double> optimum_vec_mpi = run_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec);

  double optimum_x_mpi = optimum_vec_mpi[0];
  double optimum_y_mpi = optimum_vec_mpi[1];
  double optimum_value_mpi = optimum_vec_mpi[2];

  double eps = 0.2;

  ASSERT_TRUE((0.0 <= optimum_x_mpi) && (optimum_x_mpi <= 10.0));
  ASSERT_TRUE((0.0 <= optimum_y_mpi) && (optimum_x_mpi <= 10.0));
  ASSERT_NEAR(optimum_value_mpi, -40.0, eps);

  if (world.rank() == 0) {
    std::vector<double> optimum_vec_seq = run_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec);

    double optimum_x_seq = optimum_vec_seq[0];
    double optimum_y_seq = optimum_vec_seq[1];
    double optimum_value_seq = optimum_vec_seq[2];

    ASSERT_EQ(optimum_x_seq, optimum_x_mpi);
    ASSERT_EQ(optimum_y_seq, optimum_y_mpi);
    ASSERT_EQ(optimum_value_seq, optimum_value_mpi);
  }
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_task_run_shifted_paraboloid_with_constraints) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str = "x^2y^0 -12x^1y^0 +x^0y^2 -4x^0y^1";  // shifted paraboloid, increases from about -40
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::string g_str1 = "x^0y^1 -2x^1y^0 -4";
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "x^0y^1 +x^1y^0 -4";
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::string g_str3 = "0.2x^1y^0 -x^0y^1 +0.4";
  std::vector<char> g_vec3(g_str3.length());
  std::copy(g_str3.begin(), g_str3.end(), g_vec3.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2, g_vec3});
  size_t g_count = g_vec.size();

  // Search areas:
  std::vector<double> areas_vec({0.0, 5.0, 0.0, 5.0});
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({250, 250});
  // Output value:
  std::vector<double> optimum_vec_mpi = run_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec);

  double optimum_x_mpi = optimum_vec_mpi[0];
  double optimum_y_mpi = optimum_vec_mpi[1];
  double optimum_value_mpi = optimum_vec_mpi[2];

  double eps = 0.2;

  ASSERT_TRUE((-2.0 <= optimum_x_mpi) && (optimum_x_mpi <= 4.0));
  ASSERT_TRUE((0.0 <= optimum_y_mpi) && (optimum_x_mpi <= 4.0));
  ASSERT_NEAR(optimum_value_mpi, -30.0, eps);

  if (world.rank() == 0) {
    std::vector<double> optimum_vec_seq = run_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec);

    double optimum_x_seq = optimum_vec_seq[0];
    double optimum_y_seq = optimum_vec_seq[1];
    double optimum_value_seq = optimum_vec_seq[2];

    ASSERT_EQ(optimum_x_seq, optimum_x_mpi);
    ASSERT_EQ(optimum_y_seq, optimum_y_mpi);
    ASSERT_EQ(optimum_value_seq, optimum_value_mpi);
  }
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_task_run_shifted_paraboloid_prime_num_steps) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str = "x^2y^0 -12x^1y^0 +x^0y^2 -4x^0y^1";  // shifted paraboloid, increases from about -40
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::string g_str1 = "x^0y^1 -2x^1y^0 -4";
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "x^0y^1 +x^1y^0 -4";
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::string g_str3 = "0.2x^1y^0 -x^0y^1 +0.4";
  std::vector<char> g_vec3(g_str3.length());
  std::copy(g_str3.begin(), g_str3.end(), g_vec3.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2, g_vec3});
  size_t g_count = g_vec.size();

  // Search areas:
  std::vector<double> areas_vec({0.0, 5.0, 0.0, 5.0});
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({251, 251});  // prime numbers
  // Output value:
  std::vector<double> optimum_vec_mpi = run_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec);

  double optimum_x_mpi = optimum_vec_mpi[0];
  double optimum_y_mpi = optimum_vec_mpi[1];
  double optimum_value_mpi = optimum_vec_mpi[2];

  double eps = 0.2;

  ASSERT_TRUE((-2.0 <= optimum_x_mpi) && (optimum_x_mpi <= 4.0));
  ASSERT_TRUE((0.0 <= optimum_y_mpi) && (optimum_x_mpi <= 4.0));
  ASSERT_NEAR(optimum_value_mpi, -30.0, eps);

  if (world.rank() == 0) {
    std::vector<double> optimum_vec_seq = run_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec);

    double optimum_x_seq = optimum_vec_seq[0];
    double optimum_y_seq = optimum_vec_seq[1];
    double optimum_value_seq = optimum_vec_seq[2];

    ASSERT_EQ(optimum_x_seq, optimum_x_mpi);
    ASSERT_EQ(optimum_y_seq, optimum_y_mpi);
    ASSERT_EQ(optimum_value_seq, optimum_value_mpi);
  }
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_task_run_large_degrees_without_constraints) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str = "x^256y^0 +x^0y^888 +x^100y^28";  // "box", increases from value 0
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::vector<std::vector<char>> g_vec;
  size_t g_count = g_vec.size();

  // Search areas:
  std::vector<double> areas_vec({-2.0, 2.0, -2.0, 2.0});
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({250, 250});
  // Output value:
  std::vector<double> optimum_vec_mpi = run_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec);

  double optimum_x_mpi = optimum_vec_mpi[0];
  double optimum_y_mpi = optimum_vec_mpi[1];
  double optimum_value_mpi = optimum_vec_mpi[2];

  double eps = 0.2;

  ASSERT_TRUE((-1.0 <= optimum_x_mpi) && (optimum_x_mpi <= 1.0));
  ASSERT_TRUE((-1.0 <= optimum_y_mpi) && (optimum_x_mpi <= 1.0));
  ASSERT_NEAR(optimum_value_mpi, 0.0, eps);

  if (world.rank() == 0) {
    std::vector<double> optimum_vec_seq = run_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec);

    double optimum_x_seq = optimum_vec_seq[0];
    double optimum_y_seq = optimum_vec_seq[1];
    double optimum_value_seq = optimum_vec_seq[2];

    ASSERT_EQ(optimum_x_seq, optimum_x_mpi);
    ASSERT_EQ(optimum_y_seq, optimum_y_mpi);
    ASSERT_EQ(optimum_value_seq, optimum_value_mpi);
  }
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_task_run_large_degrees_with_constraints) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str = "x^256y^0 +x^0y^888 +x^100y^28";  // "box", increases from value 0
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::string g_str1 = "-x^1y^0 +0.5";  // x >= 0.5
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1 +0.7";  // y >= 0.7
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  size_t g_count = g_vec.size();

  // Search areas:
  std::vector<double> areas_vec({-2.0, 2.0, -2.0, 2.0});
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({250, 250});
  // Output value:
  std::vector<double> optimum_vec_mpi = run_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec);

  double optimum_x_mpi = optimum_vec_mpi[0];
  double optimum_y_mpi = optimum_vec_mpi[1];
  double optimum_value_mpi = optimum_vec_mpi[2];

  double eps = 0.2;

  ASSERT_TRUE(0.5 <= optimum_x_mpi);
  ASSERT_TRUE(0.7 <= optimum_y_mpi);
  ASSERT_NEAR(optimum_value_mpi, 0.0, eps);

  if (world.rank() == 0) {
    std::vector<double> optimum_vec_seq = run_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec);

    double optimum_x_seq = optimum_vec_seq[0];
    double optimum_y_seq = optimum_vec_seq[1];
    double optimum_value_seq = optimum_vec_seq[2];

    ASSERT_EQ(optimum_x_seq, optimum_x_mpi);
    ASSERT_EQ(optimum_y_seq, optimum_y_mpi);
    ASSERT_EQ(optimum_value_seq, optimum_value_mpi);
  }
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_task_run_large_degrees_degree2_prime_num) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str = "x^256y^0 +x^0y^888 +x^100y^28";  // "box", increases from value 0
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::string g_str1 = "-x^1y^0 +0.5";  // x >= 0.5
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1 +0.7";  // y >= 0.7
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  size_t g_count = g_vec.size();

  // Search areas:
  std::vector<double> areas_vec({-2.0, 2.0, -2.0, 2.0});
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({256, 251});  // 2^8 and prime number
  // Output value:
  std::vector<double> optimum_vec_mpi = run_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec);

  double optimum_x_mpi = optimum_vec_mpi[0];
  double optimum_y_mpi = optimum_vec_mpi[1];
  double optimum_value_mpi = optimum_vec_mpi[2];

  double eps = 0.2;

  ASSERT_TRUE(0.5 <= optimum_x_mpi);
  ASSERT_TRUE(0.7 <= optimum_y_mpi);
  ASSERT_NEAR(optimum_value_mpi, 0.0, eps);

  if (world.rank() == 0) {
    std::vector<double> optimum_vec_seq = run_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec);

    double optimum_x_seq = optimum_vec_seq[0];
    double optimum_y_seq = optimum_vec_seq[1];
    double optimum_value_seq = optimum_vec_seq[2];

    ASSERT_EQ(optimum_x_seq, optimum_x_mpi);
    ASSERT_EQ(optimum_y_seq, optimum_y_mpi);
    ASSERT_EQ(optimum_value_seq, optimum_value_mpi);
  }
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_task_run_odd_degrees_without_constraints) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str = "x^33y^0 +x^0y^51";  // "increasing-decreasing box", increases from 0 where x > ~0.75 or y > ~0.75
                                           // decreases from 0 where x < ~-0.75 or y < ~-0.75
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::vector<std::vector<char>> g_vec;
  size_t g_count = g_vec.size();

  // Search areas:
  std::vector<double> areas_vec({0.0, 2.0, 0.0, 2.0});  // areas changed to find min!!!
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({250, 250});
  // Output value:
  std::vector<double> optimum_vec_mpi = run_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec);

  double optimum_x_mpi = optimum_vec_mpi[0];
  double optimum_y_mpi = optimum_vec_mpi[1];
  double optimum_value_mpi = optimum_vec_mpi[2];

  double eps = 0.2;

  ASSERT_TRUE((0.0 <= optimum_x_mpi) && (optimum_x_mpi <= 1.0));
  ASSERT_TRUE((0.0 <= optimum_y_mpi) && (optimum_x_mpi <= 1.0));
  ASSERT_NEAR(optimum_value_mpi, 0.0, eps);

  if (world.rank() == 0) {
    std::vector<double> optimum_vec_seq = run_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec);

    double optimum_x_seq = optimum_vec_seq[0];
    double optimum_y_seq = optimum_vec_seq[1];
    double optimum_value_seq = optimum_vec_seq[2];

    ASSERT_EQ(optimum_x_seq, optimum_x_mpi);
    ASSERT_EQ(optimum_y_seq, optimum_y_mpi);
    ASSERT_EQ(optimum_value_seq, optimum_value_mpi);
  }
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_task_run_odd_degrees_with_constraints) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str = "x^33y^0 +x^0y^51";  // "increasing-decreasing box", increases from 0 where x > ~0.75 or y > ~0.75
                                           // decreases from 0 where x < ~-0.75 or y < ~-0.75
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::string g_str1 = "-x^1y^0 -0.5";  // x >= -0.5
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1 -0.5";  // y >= -0.5
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  size_t g_count = g_vec.size();

  // Search areas:
  std::vector<double> areas_vec({-2.0, 2.0, -2.0, 2.0});
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({250, 250});
  // Output value:
  std::vector<double> optimum_vec_mpi = run_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec);

  double optimum_x_mpi = optimum_vec_mpi[0];
  double optimum_y_mpi = optimum_vec_mpi[1];
  double optimum_value_mpi = optimum_vec_mpi[2];

  double eps = 0.2;

  ASSERT_TRUE((-0.5 <= optimum_x_mpi) && (optimum_x_mpi <= 1.0));
  ASSERT_TRUE((-0.5 <= optimum_y_mpi) && (optimum_x_mpi <= 1.0));
  ASSERT_NEAR(optimum_value_mpi, 0.0, eps);

  if (world.rank() == 0) {
    std::vector<double> optimum_vec_seq = run_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec);

    double optimum_x_seq = optimum_vec_seq[0];
    double optimum_y_seq = optimum_vec_seq[1];
    double optimum_value_seq = optimum_vec_seq[2];

    ASSERT_EQ(optimum_x_seq, optimum_x_mpi);
    ASSERT_EQ(optimum_y_seq, optimum_y_mpi);
    ASSERT_EQ(optimum_value_seq, optimum_value_mpi);
  }
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_task_run_negative_degree_without_constraints) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str = "x^-2y^0";  // "3D hyperbole" with positive values
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::vector<std::vector<char>> g_vec;
  size_t g_count = g_vec.size();

  // Search areas:
  std::vector<double> areas_vec({-15.0, 0.0, -10.0, 10.0});  // areas changed to find min!!!
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({250, 250});
  // Output value:
  std::vector<double> optimum_vec_mpi = run_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec);

  double optimum_value_mpi = optimum_vec_mpi[2];

  double eps = 0.2;

  ASSERT_NEAR(optimum_value_mpi, 0.0, eps);

  if (world.rank() == 0) {
    std::vector<double> optimum_vec_seq = run_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec);

    double optimum_value_seq = optimum_vec_seq[2];

    ASSERT_EQ(optimum_value_seq, optimum_value_mpi);
  }
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_func, test_task_run_negative_degree_with_constraints) {
  boost::mpi::communicator world;

  // Criterium-function:
  std::string q_str = "x^-2y^0";  // "3D hyperbole" with positive values
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  // Constraints-functions:
  std::string g_str1 = "-x^1y^0 -10";  // x >= -10
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1 -0.5";  // y >= -0.5
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::string g_str3 = "x^0y^1 -0.5";  // y <= 0.5
  std::vector<char> g_vec3(g_str3.length());
  std::copy(g_str3.begin(), g_str3.end(), g_vec3.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2, g_vec3});
  size_t g_count = g_vec.size();

  // Search areas:
  std::vector<double> areas_vec({-15.0, 0.0, -10.0, 10.0});  // areas changed to find min!!!
  // Steps counts (how many points will be used):
  std::vector<int> steps_vec({250, 250});
  // Output value:
  std::vector<double> optimum_vec_mpi = run_test_mpi(q_vec, g_count, g_vec, areas_vec, steps_vec);

  double optimum_x_mpi = optimum_vec_mpi[0];
  double optimum_y_mpi = optimum_vec_mpi[1];
  double optimum_value_mpi = optimum_vec_mpi[2];

  double eps = 0.2;

  ASSERT_TRUE(-10.0 <= optimum_x_mpi);
  ASSERT_TRUE((-0.5 <= optimum_y_mpi) && (optimum_x_mpi <= 0.5));
  ASSERT_NEAR(optimum_value_mpi, 0.0, eps);

  if (world.rank() == 0) {
    std::vector<double> optimum_vec_seq = run_test_seq(q_vec, g_count, g_vec, areas_vec, steps_vec);

    double optimum_x_seq = optimum_vec_seq[0];
    double optimum_y_seq = optimum_vec_seq[1];
    double optimum_value_seq = optimum_vec_seq[2];

    ASSERT_EQ(optimum_x_seq, optimum_x_mpi);
    ASSERT_EQ(optimum_y_seq, optimum_y_mpi);
    ASSERT_EQ(optimum_value_seq, optimum_value_mpi);
  }
}
