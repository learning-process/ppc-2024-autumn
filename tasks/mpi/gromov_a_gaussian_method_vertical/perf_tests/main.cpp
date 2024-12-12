#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <functional>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/gromov_a_gaussian_method_vertical/include/ops_mpi.hpp"

namespace gromov_a_gaussian_method_vertical_mpi {
std::vector<int> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(-100, 100);
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen();
  }
  return vec;
}
}  // namespace gromov_a_gaussian_method_vertical_mpi

TEST(gromov_a_gaussian_method_vertical_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int equations = 800;
  int size_coefficient_mat = equations * equations;
  int band_width = 15;
  std::vector<int> input_coefficient(size_coefficient_mat, 0);
  std::vector<int> input_rhs(equations, 0);
  std::vector<double> func_res(equations, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input_rhs = gromov_a_gaussian_method_vertical_mpi::getRandomVector(equations);
    input_coefficient = gromov_a_gaussian_method_vertical_mpi::getRandomVector(size_coefficient_mat);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coefficient.data()));
    taskDataPar->inputs_count.emplace_back(input_coefficient.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_rhs.data()));
    taskDataPar->inputs_count.emplace_back(input_rhs.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
    taskDataPar->outputs_count.emplace_back(func_res.size());
  }

  auto MPIGaussVerticalParallel =
      std::make_shared<gromov_a_gaussian_method_vertical_mpi::MPIGaussVerticalParallel>(taskDataPar, band_width);
  ASSERT_EQ(MPIGaussVerticalParallel->validation(), true);
  MPIGaussVerticalParallel->pre_processing();
  MPIGaussVerticalParallel->run();
  MPIGaussVerticalParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MPIGaussVerticalParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(int(func_res.size()), equations);
  }
}

TEST(gromov_a_gaussian_method_vertical_mpi, test_task_run) {
  boost::mpi::communicator world;
  int equations = 800;
  int size_coefficient_mat = equations * equations;
  int band_width = 15;
  std::vector<int> input_coefficient(size_coefficient_mat, 0);
  std::vector<int> input_rhs(equations, 0);
  std::vector<double> func_res(equations, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input_rhs = gromov_a_gaussian_method_vertical_mpi::getRandomVector(equations);
    input_coefficient = gromov_a_gaussian_method_vertical_mpi::getRandomVector(size_coefficient_mat);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coefficient.data()));
    taskDataPar->inputs_count.emplace_back(input_coefficient.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_rhs.data()));
    taskDataPar->inputs_count.emplace_back(input_rhs.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
    taskDataPar->outputs_count.emplace_back(func_res.size());
  }

  auto MPIGaussVerticalParallel =
      std::make_shared<gromov_a_gaussian_method_vertical_mpi::MPIGaussVerticalParallel>(taskDataPar, band_width);
  ASSERT_EQ(MPIGaussVerticalParallel->validation(), true);
  MPIGaussVerticalParallel->pre_processing();
  MPIGaussVerticalParallel->run();
  MPIGaussVerticalParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MPIGaussVerticalParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(int(func_res.size()), equations);
  }
}
