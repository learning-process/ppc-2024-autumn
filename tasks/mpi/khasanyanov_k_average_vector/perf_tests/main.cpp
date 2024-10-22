#include <boost/mpi/timer.hpp>
#include <chrono>
#include <thread>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/khasanyanov_k_average_vector/include/avg_mpi.hpp"

//=========================================sequence=========================================

const int SIZE = 1220000;

TEST(khasanyanov_k_average_vector_seq, test_pipeline_run) {
  std::vector<int> global_vec(SIZE, 4);
  std::vector<double> average(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData =
      khasanyanov_k_average_vector_mpi::create_task_data<int, double>(global_vec, average);

  auto testAvgVectorSequence =
      std::make_shared<khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<int, double>>(taskData);

  RUN_TASK(*testAvgVectorSequence);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testAvgVectorSequence);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  EXPECT_NEAR(4, average[0], 1e-5);
}

TEST(khasanyanov_k_average_vector_seq, test_task_run) {
  std::vector<int> global_vec(SIZE, 4);
  std::vector<double> average(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData =
      khasanyanov_k_average_vector_mpi::create_task_data<int, double>(global_vec, average);

  auto testAvgVectorSequence =
      std::make_shared<khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<int, double>>(taskData);

  RUN_TASK(*testAvgVectorSequence);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testAvgVectorSequence);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  EXPECT_NEAR(4, average[0], 1e-5);
}

//=========================================parallel=========================================

TEST(khasanyanov_k_average_vector_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<double> global_vec;
  std::vector<double> average_par(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = khasanyanov_k_average_vector_mpi::get_random_vector<double>(SIZE);
    taskDataPar = khasanyanov_k_average_vector_mpi::create_task_data<double, double>(global_vec, average_par);
  }

  auto testMpiTaskParallel =
      std::make_shared<khasanyanov_k_average_vector_mpi::AvgVectorMPITaskParallel<double, double>>(taskDataPar);

  RUN_TASK(*testMpiTaskParallel);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    std::vector<double> average_seq(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq =
        khasanyanov_k_average_vector_mpi::create_task_data<double, double>(global_vec, average_seq);
    auto testMpiTaskSequential =
        khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<double, double>(taskDataSeq);
    RUN_TASK(testMpiTaskSequential);
    ppc::core::Perf::print_perf_statistic(perfResults);
    EXPECT_NEAR(average_seq[0], average_par[0], 1e-5);
  }
}

TEST(khasanyanov_k_average_vector_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<double> global_vec;
  std::vector<double> average_par(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = khasanyanov_k_average_vector_mpi::get_random_vector<double>(SIZE);
    taskDataPar = khasanyanov_k_average_vector_mpi::create_task_data<double, double>(global_vec, average_par);
  }

  auto testMpiTaskParallel =
      std::make_shared<khasanyanov_k_average_vector_mpi::AvgVectorMPITaskParallel<double, double>>(taskDataPar);

  RUN_TASK(*testMpiTaskParallel);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    std::vector<double> average_seq(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq =
        khasanyanov_k_average_vector_mpi::create_task_data<double, double>(global_vec, average_seq);
    auto testMpiTaskSequential =
        khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<double, double>(taskDataSeq);
    RUN_TASK(testMpiTaskSequential);
    ppc::core::Perf::print_perf_statistic(perfResults);
    EXPECT_NEAR(average_seq[0], average_par[0], 1e-5);
  }
}