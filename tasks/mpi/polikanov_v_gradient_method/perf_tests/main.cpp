#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <cmath>

#include "core/perf/include/perf.hpp"
#include "mpi/polikanov_v_gradient_method/include/ops_mpi.hpp"

namespace polikanov_v_gradient_method_mpi {

std::vector<double> GenerateLargeSymmetricMatrix(int size) {
  std::vector<double> matrix(size * size, 0.0);

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      if (i == j) {
        matrix[i * size + j] = size + 1.0;
      } else {
        matrix[i * size + j] = 1.0;
      }
    }
  }

  return matrix;
}

}  // namespace polikanov_v_gradient_method_mpi

TEST(polikanov_v_gradient_method_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int size = std::pow(2, 10);
  std::vector<double> flat_matrix = polikanov_v_gradient_method_mpi::GenerateLargeSymmetricMatrix(size);
  std::vector<double> rhs(size, 1.0);
  std::vector<double> initialGuess(size, 0.0);
  std::vector<double> expected(size, 0.00048804);
  double tolerance = 1e-6;
  std::vector<double> result(size);
  std::shared_ptr<ppc::core::TaskData> task = std::make_shared<ppc::core::TaskData>();
  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(&size));
  task->inputs_count.emplace_back(1);
  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(&tolerance));
  task->inputs_count.emplace_back(1);
  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(flat_matrix.data()));
  task->inputs_count.emplace_back(flat_matrix.size());
  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(rhs.data()));
  task->inputs_count.emplace_back(rhs.size());
  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(initialGuess.data()));
  task->inputs_count.emplace_back(initialGuess.size());
  task->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  auto GradientMethod = std::make_shared<polikanov_v_gradient_method_mpi::GradientMethod>(task);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(GradientMethod);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) ppc::core::Perf::print_perf_statistic(perfResults);
  for (size_t i = 0; i < expected.size(); i++) ASSERT_NEAR(result[i], expected[i], tolerance);
}

TEST(polikanov_v_gradient_method_mpi, test_task_run) {
  boost::mpi::communicator world;
  int size = std::pow(2, 10);
  std::vector<double> flat_matrix = polikanov_v_gradient_method_mpi::GenerateLargeSymmetricMatrix(size);
  std::vector<double> rhs(size, 1.0);
  std::vector<double> initialGuess(size, 0.0);
  std::vector<double> expected(size, 0.00048804);
  double tolerance = 1e-6;
  std::vector<double> result(size);
  std::shared_ptr<ppc::core::TaskData> task = std::make_shared<ppc::core::TaskData>();
  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(&size));
  task->inputs_count.emplace_back(1);
  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(&tolerance));
  task->inputs_count.emplace_back(1);
  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(flat_matrix.data()));
  task->inputs_count.emplace_back(flat_matrix.size());
  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(rhs.data()));
  task->inputs_count.emplace_back(rhs.size());
  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(initialGuess.data()));
  task->inputs_count.emplace_back(initialGuess.size());
  task->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  auto GradientMethod = std::make_shared<polikanov_v_gradient_method_mpi::GradientMethod>(task);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(GradientMethod);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) ppc::core::Perf::print_perf_statistic(perfResults);
  for (size_t i = 0; i < expected.size(); i++) ASSERT_NEAR(result[i], expected[i], tolerance);
}
