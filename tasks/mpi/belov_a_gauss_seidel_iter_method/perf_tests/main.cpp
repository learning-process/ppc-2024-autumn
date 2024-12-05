#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/belov_a_gauss_seidel_iter_method/include/ops_mpi.hpp"

using namespace belov_a_gauss_seidel_mpi;

TEST(belov_a_gauss_seidel_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;

  int n = 1000;
  double epsilon = 0.001;
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
  double epsilon = 0.001;
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