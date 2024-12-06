#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/veliev_e_iterative_methods_jacobi/include/ops_seq.hpp"

TEST(veliev_e_iterative_methods_jacobi, pipeline_solver_test) {
  const int grid_size = 300;
  std::vector<double> coefficient_matrix(grid_size * grid_size, 1.5);
  for (int i = 0; i < grid_size; ++i) {
    coefficient_matrix[i * grid_size + i] = 450.0;
  }
  std::vector<double> rhs_values(grid_size, 450.0);
  std::vector<double> computed_solution(grid_size, 0.0);
  std::vector<double> reference_solution(grid_size, 1.0);

  auto data_package = std::make_shared<ppc::core::TaskData>();
  data_package->inputs.emplace_back(reinterpret_cast<uint8_t *>(coefficient_matrix.data()));
  data_package->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhs_values.data()));
  data_package->inputs.emplace_back(reinterpret_cast<uint8_t *>(computed_solution.data()));
  data_package->inputs_count.emplace_back(grid_size);
  data_package->inputs_count.emplace_back(rhs_values.size());
  data_package->inputs_count.emplace_back(computed_solution.size());
  data_package->outputs.emplace_back(reinterpret_cast<uint8_t *>(computed_solution.data()));
  data_package->outputs_count.emplace_back(computed_solution.size());

  auto jacobi_solver = std::make_shared<veliev_e_iterative_methods_jacobi::MethodJacobi>(data_package);

  auto execution_attributes = std::make_shared<ppc::core::PerfAttr>();
  execution_attributes->num_running = 5;
  const auto start_time = std::chrono::high_resolution_clock::now();
  execution_attributes->current_timer = [&] {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - start_time).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto execution_metrics = std::make_shared<ppc::core::PerfResults>();

  auto pipeline_executor = std::make_shared<ppc::core::Perf>(jacobi_solver);
  pipeline_executor->pipeline_run(execution_attributes, execution_metrics);
  ppc::core::Perf::print_perf_statistic(execution_metrics);

  for (int i = 0; i < grid_size; ++i) {
    ASSERT_NEAR(computed_solution[i], reference_solution[i], 0.5);
  }
}

TEST(veliev_e_iterative_methods_jacobi, task_solver_test) {
  const int grid_size = 300;
  std::vector<double> coefficient_matrix(grid_size * grid_size, 1.5);
  for (int i = 0; i < grid_size; ++i) {
    coefficient_matrix[i * grid_size + i] = 450.0;
  }
  std::vector<double> rhs_values(grid_size, 450.0);
  std::vector<double> computed_solution(grid_size, 0.0);
  std::vector<double> reference_solution(grid_size, 1.0);

  auto data_package = std::make_shared<ppc::core::TaskData>();
  data_package->inputs.emplace_back(reinterpret_cast<uint8_t *>(coefficient_matrix.data()));
  data_package->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhs_values.data()));
  data_package->inputs.emplace_back(reinterpret_cast<uint8_t *>(computed_solution.data()));
  data_package->inputs_count.emplace_back(grid_size);
  data_package->inputs_count.emplace_back(rhs_values.size());
  data_package->inputs_count.emplace_back(computed_solution.size());
  data_package->outputs.emplace_back(reinterpret_cast<uint8_t *>(computed_solution.data()));
  data_package->outputs_count.emplace_back(computed_solution.size());

  auto jacobi_solver = std::make_shared<veliev_e_iterative_methods_jacobi::MethodJacobi>(data_package);

  auto execution_attributes = std::make_shared<ppc::core::PerfAttr>();
  execution_attributes->num_running = 5;
  const auto start_time = std::chrono::high_resolution_clock::now();
  execution_attributes->current_timer = [&] {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - start_time).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto execution_metrics = std::make_shared<ppc::core::PerfResults>();

  auto standalone_executor = std::make_shared<ppc::core::Perf>(jacobi_solver);
  standalone_executor->task_run(execution_attributes, execution_metrics);
  ppc::core::Perf::print_perf_statistic(execution_metrics);

  for (int i = 0; i < grid_size; ++i) {
    ASSERT_NEAR(computed_solution[i], reference_solution[i], 0.5);
  }
}
