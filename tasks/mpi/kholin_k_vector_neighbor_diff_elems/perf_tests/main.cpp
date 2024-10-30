#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kholin_k_vector_neighbor_diff_elems/include/ops_mpi.hpp"

TEST(kholin_k_vector_neighbor_diff_elems_mpi, test_pipeline_run) {
  int ProcRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  std::vector<int> global_vec;
  std::vector<int> global_elems(2, 0);
  std::vector<uint64_t> global_indices(2, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (ProcRank == 0) {
    const float count_size_vector = 100000000;
    global_vec = std::vector<int>(count_size_vector);
    for (size_t i = 0; i < global_vec.size(); i++) {
      global_vec[i] = 4 * i + 2;
    }

    global_vec[10] = 5000;
    global_vec[11] = 1;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_elems.data()));
    taskDataPar->outputs_count.emplace_back(global_elems.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_indices.data()));
    taskDataPar->outputs_count.emplace_back(global_indices.size());
  }

  auto testMpiTaskParallel = std::make_shared<kholin_k_vector_neighbor_diff_elems_mpi::TestMPITaskParallel<int>>(
      taskDataPar, "MAX_DIFFERENCE");
  testMpiTaskParallel->validation();
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
  if (ProcRank == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(kholin_k_vector_neighbor_diff_elems_mpi, test_task_run) {
  int ProcRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  std::vector<int> global_vec;
  std::vector<int> global_elems(2, 0);
  std::vector<uint64_t> global_indices(2, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (ProcRank == 0) {
    const float count_size_vector = 100000000;
    global_vec = std::vector<int>(count_size_vector);
    for (size_t i = 0; i < global_vec.size(); i++) {
      global_vec[i] = 4 * i + 2;
    }

    global_vec[10] = 5000;
    global_vec[11] = 1;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_elems.data()));
    taskDataPar->outputs_count.emplace_back(global_elems.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_indices.data()));
    taskDataPar->outputs_count.emplace_back(global_indices.size());
  }

  auto testMpiTaskParallel = std::make_shared<kholin_k_vector_neighbor_diff_elems_mpi::TestMPITaskParallel<int>>(
      taskDataPar, "MAX_DIFFERENCE");
  testMpiTaskParallel->validation();
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
  if (ProcRank == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}
