#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kavtorev_d_iterative_jacobi/include/ops_mpi.hpp"

namespace kavtorev_d_iterative_jacobi_mpi {

void generateTestData(size_t n, std::vector<double>& A_flat, std::vector<double>& F) {
  std::vector<double> x;
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> offDiagDist(-10.0, 10.0);
  std::uniform_real_distribution<double> diagDist(20.0, 50.0);
  std::uniform_real_distribution<double> solDist(-100.0, 100.0);

  x.resize(n);
  for (size_t i = 0; i < n; ++i) {
    x[i] = solDist(gen);
  }

  A_flat.resize(n * n, 0.0);

  for (size_t i = 0; i < n; ++i) {
    double sum = 0.0;

    for (size_t j = 0; j < n; ++j) {
      if (i != j) {
        double value = offDiagDist(gen);
        A_flat[i * n + j] = value;
        sum += std::abs(value);
      }
    }

    A_flat[i * n + i] = sum + diagDist(gen);
  }

  F.resize(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    double sum = 0.0;
    for (size_t j = 0; j < n; ++j) {
      sum += A_flat[i * n + j] * x[j];
    }
    F[i] = sum;
  }
}

}  // namespace kavtorev_d_iterative_jacobi_mpi

TEST(kavtorev_d_iterative_jacobi_mpi, pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  int n = 1001;
  double eps = 0.001;
  int iterations = 1000;

  std::vector<double> A;
  std::vector<double> F;
  std::vector<double> mpi_X;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
  taskDataPar->inputs_count.emplace_back(1);

  if (world.rank() == 0) {
    kavtorev_d_iterative_jacobi_mpi::generateTestData(n, A, F);
    mpi_X.resize(n);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(F.data()));
    taskDataPar->inputs_count.emplace_back(F.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(mpi_X.data()));
    taskDataPar->outputs_count.emplace_back(mpi_X.size());
  }

  auto taskParallel = std::make_shared<kavtorev_d_iterative_jacobi_mpi::IterativeJacobiParallelMPI>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  bool mpi_run_res = taskParallel->run();
  taskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    if (mpi_run_res) {
      std::vector<double> seq_X(n);

      std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
      taskDataSeq->inputs_count.emplace_back(1);

      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
      taskDataSeq->inputs_count.emplace_back(1);

      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
      taskDataSeq->inputs_count.emplace_back(1);

      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
      taskDataSeq->inputs_count.emplace_back(A.size());

      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(F.data()));
      taskDataSeq->inputs_count.emplace_back(F.size());

      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_X.data()));
      taskDataSeq->outputs_count.emplace_back(seq_X.size());

      auto taskSequential =
          std::make_shared<kavtorev_d_iterative_jacobi_mpi::IterativeJacobiSequentialMPI>(taskDataSeq);
      ASSERT_TRUE(taskSequential->validation());
      taskSequential->pre_processing();
      ASSERT_TRUE(taskSequential->run());
      taskSequential->post_processing();

      ASSERT_EQ(seq_X.size(), mpi_X.size());
      double error = 0.0;
      for (int i = 0; i < n; ++i) {
        error = std::max(error, std::abs(seq_X[i] - mpi_X[i]));
      }
      EXPECT_LT(error, 2 * eps);
    }
  }
}

TEST(kavtorev_d_iterative_jacobi_mpi, task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  int n = 1000;
  double eps = 0.001;
  int iterations = 1000;

  std::vector<double> A;
  std::vector<double> F;
  std::vector<double> mpi_X;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
  taskDataPar->inputs_count.emplace_back(1);

  if (world.rank() == 0) {
    kavtorev_d_iterative_jacobi_mpi::generateTestData(n, A, F);
    mpi_X.resize(n);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(F.data()));
    taskDataPar->inputs_count.emplace_back(F.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(mpi_X.data()));
    taskDataPar->outputs_count.emplace_back(mpi_X.size());
  }

  auto taskParallel = std::make_shared<kavtorev_d_iterative_jacobi_mpi::IterativeJacobiParallelMPI>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  bool mpi_run_res = taskParallel->run();
  taskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    if (mpi_run_res) {
      std::vector<double> seq_X(n);

      std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
      taskDataSeq->inputs_count.emplace_back(1);

      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
      taskDataSeq->inputs_count.emplace_back(1);

      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
      taskDataSeq->inputs_count.emplace_back(1);

      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
      taskDataSeq->inputs_count.emplace_back(A.size());

      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(F.data()));
      taskDataSeq->inputs_count.emplace_back(F.size());

      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_X.data()));
      taskDataSeq->outputs_count.emplace_back(seq_X.size());

      auto taskSequential =
          std::make_shared<kavtorev_d_iterative_jacobi_mpi::IterativeJacobiSequentialMPI>(taskDataSeq);
      ASSERT_TRUE(taskSequential->validation());
      taskSequential->pre_processing();
      ASSERT_TRUE(taskSequential->run());
      taskSequential->post_processing();

      ASSERT_EQ(seq_X.size(), mpi_X.size());
      double error = 0.0;
      for (int i = 0; i < n; ++i) {
        error = std::max(error, std::abs(seq_X[i] - mpi_X[i]));
      }
      EXPECT_LT(error, 2 * eps);
    }
  }
}
