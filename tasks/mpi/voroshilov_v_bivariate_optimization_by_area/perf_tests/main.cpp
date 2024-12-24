#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/voroshilov_v_bivariate_optimization_by_area/include/ops_mpi.hpp"

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_perf, test_pipeline_run) {
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
  std::vector<int> steps_vec({2000, 2000});

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
  if (world.rank() == 0) {
    // Search area boundaries:
    taskDataParallel->inputs_count.emplace_back(areas_vec.size());
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(areas_vec.data()));
    // Steps counts (how many points will be used):
    taskDataParallel->inputs_count.emplace_back(steps_vec.size());
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(steps_vec.data()));
    // Output - optimum point and value:
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(optimum_vec.data()));
  }

  auto optimizationMPITaskParallel =
      std::make_shared<voroshilov_v_bivariate_optimization_by_area_mpi::OptimizationMPITaskParallel>(taskDataParallel);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(optimizationMPITaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    double optimum_x_mpi = optimum_vec[0];
    double optimum_y_mpi = optimum_vec[1];
    double optimum_value_mpi = optimum_vec[2];

    double eps = 0.2;

    ASSERT_NEAR(optimum_x_mpi, 1.0, eps);
    ASSERT_NEAR(optimum_y_mpi, 1.0, eps);
    ASSERT_NEAR(optimum_value_mpi, 2.0, eps);
  }
}

TEST(voroshilov_v_bivariate_optimization_by_area_mpi_perf, test_task_run) {
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
  std::vector<int> steps_vec({2000, 2000});

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
  if (world.rank() == 0) {
    // Search area boundaries:
    taskDataParallel->inputs_count.emplace_back(areas_vec.size());
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(areas_vec.data()));
    // Steps counts (how many points will be used):
    taskDataParallel->inputs_count.emplace_back(steps_vec.size());
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(steps_vec.data()));
    // Output - optimum point and value:
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(optimum_vec.data()));
  }

  auto optimizationMPITaskParallel =
      std::make_shared<voroshilov_v_bivariate_optimization_by_area_mpi::OptimizationMPITaskParallel>(taskDataParallel);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(optimizationMPITaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    double optimum_x_mpi = optimum_vec[0];
    double optimum_y_mpi = optimum_vec[1];
    double optimum_value_mpi = optimum_vec[2];

    double eps = 0.2;

    ASSERT_NEAR(optimum_x_mpi, 1.0, eps);
    ASSERT_NEAR(optimum_y_mpi, 1.0, eps);
    ASSERT_NEAR(optimum_value_mpi, 2.0, eps);
  }
}
