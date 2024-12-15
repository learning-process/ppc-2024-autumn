#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/belov_a_gauss_seidel_iter_method/include/ops_mpi.hpp"

using namespace belov_a_gauss_seidel_mpi;

namespace belov_a_gauss_seidel_mpi {
std::vector<double> generateDiagonallyDominantMatrix(int n) {
  std::vector<double> A_local(n * n, 0.0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-10.0, 10.0);

  for (int i = 0; i < n; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < n; ++j) {
      if (i != j) {
        A_local[i * n + j] = dis(gen);
        row_sum += abs(A_local[i * n + j]);
      }
    }
    A_local[i * n + i] = row_sum + abs(dis(gen)) + 1.0;
  }
  return A_local;
}

std::vector<double> generateFreeMembers(int n) {
  std::vector<double> freeMembers(n, 0.0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-10.0, 10.0);

  for (int i = 0; i < n; ++i) {
    freeMembers[i] = dis(gen);
  }
  return freeMembers;
}
}  // namespace belov_a_gauss_seidel_mpi

TEST(belov_a_gauss_seidel_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;

  int n = 1000;
  double epsilon = 0.2;
  std::vector<double> matrix = generateDiagonallyDominantMatrix(n);
  std::vector<double> freeMembers = generateFreeMembers(n);
  std::vector<double> solutionMpi(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs_count.emplace_back(freeMembers.size());
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionMpi.data()));
    taskDataPar->outputs_count.emplace_back(solutionMpi.size());
  }

  // Create Task
  auto testMpiTaskParallel = std::make_shared<belov_a_gauss_seidel_mpi::GaussSeidelParallel>(taskDataPar);

  ASSERT_TRUE(testMpiTaskParallel->validation());
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    std::vector<double> solutionSeq(n, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs_count.emplace_back(freeMembers.size());
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionSeq.data()));
    taskDataSeq->outputs_count.emplace_back(solutionSeq.size());

    auto testMpiTaskSequential = std::make_shared<belov_a_gauss_seidel_mpi::GaussSeidelSequential>(taskDataSeq);

    ASSERT_TRUE(testMpiTaskSequential->validation());
    testMpiTaskSequential->pre_processing();
    testMpiTaskSequential->run();
    testMpiTaskSequential->post_processing();

    ppc::core::Perf::print_perf_statistic(perfResults);

    for (int i = 0; i < n; ++i) {
      ASSERT_NEAR(solutionMpi[i], solutionSeq[i], epsilon);
    }
  }
}

TEST(belov_a_gauss_seidel_perf_test, test_task_run) {
  boost::mpi::communicator world;

  int n = 1000;
  double epsilon = 0.2;
  std::vector<double> matrix = generateDiagonallyDominantMatrix(n);
  std::vector<double> freeMembers = generateFreeMembers(n);
  std::vector<double> solutionMpi(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs_count.emplace_back(freeMembers.size());
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionMpi.data()));
    taskDataPar->outputs_count.emplace_back(solutionMpi.size());
  }

  // Create Task
  auto testMpiTaskParallel = std::make_shared<belov_a_gauss_seidel_mpi::GaussSeidelParallel>(taskDataPar);

  ASSERT_TRUE(testMpiTaskParallel->validation());
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    std::vector<double> solutionSeq(n, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs_count.emplace_back(freeMembers.size());
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionSeq.data()));
    taskDataSeq->outputs_count.emplace_back(solutionSeq.size());

    auto testMpiTaskSequential = std::make_shared<belov_a_gauss_seidel_mpi::GaussSeidelSequential>(taskDataSeq);

    ASSERT_TRUE(testMpiTaskSequential->validation());
    testMpiTaskSequential->pre_processing();
    testMpiTaskSequential->run();
    testMpiTaskSequential->post_processing();

    ppc::core::Perf::print_perf_statistic(perfResults);

    for (int i = 0; i < n; ++i) {
      ASSERT_NEAR(solutionMpi[i], solutionSeq[i], epsilon);
    }
  }
}