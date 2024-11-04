// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/naumov_b_min_colum_matrix/include/ops_mpi.hpp"

// // Generates random string with given size filled with digits 0-9
// std::vector<int> naumov_b_min_colum_matrix_mpi::getRandomVector(int size) {
//   std::vector<int> vec(size);
//   for (int &element : vec) {
//     element = rand() % 201 - 100;
//   }
//   return vec;
// }

// TEST(naumov_b_min_colum_matrix_mpi_perf_test, test_pipeline_run) {
//   boost::mpi::communicator world;
//   const int rows = 100;
//   const int cols = 100;
//   std::vector<int> global_matrix;
//   std::vector<int> global_minima(cols, std::numeric_limits<int>::max());

//   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

//   if (world.rank() == 0) {
//     global_matrix = naumov_b_min_colum_matrix_mpi::getRandomVector(cols * rows);

//     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
//     taskDataPar->inputs_count.emplace_back(rows);
//     taskDataPar->inputs_count.emplace_back(cols);
//     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_minima.data()));
//     taskDataPar->outputs_count.emplace_back(global_minima.size());
//   }

//   auto testMpiTaskParallel = std::make_shared<naumov_b_min_colum_matrix_mpi::TestMPITaskParallel>(taskDataPar);
//   ASSERT_TRUE(testMpiTaskParallel->validation());
//   testMpiTaskParallel->pre_processing();
//   testMpiTaskParallel->run();
//   testMpiTaskParallel->post_processing();

//   // Create Perf attributes
//   auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
//   perfAttr->num_running = 10;
//   const boost::mpi::timer current_timer;
//   perfAttr->current_timer = [&] { return current_timer.elapsed(); };

//   // Create and init perf results
//   auto perfResults = std::make_shared<ppc::core::PerfResults>();

//   // Create Perf analyzer
//   auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
//   perfAnalyzer->pipeline_run(perfAttr, perfResults);
//   if (world.rank() == 0) {
//     ppc::core::Perf::print_perf_statistic(perfResults);
//   }
// }

// TEST(naumov_b_min_colum_matrix_mpi_perf_test, test_task_run) {
//   boost::mpi::communicator world;
//   const int rows = 10;
//   const int cols = 40;
//   std::vector<int> global_matrix;
//   std::vector<int> global_minima(cols, std::numeric_limits<int>::max());

//   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

//   if (world.rank() == 0) {
//     global_matrix = naumov_b_min_colum_matrix_mpi::getRandomVector(cols * rows);

//     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
//     taskDataPar->inputs_count.emplace_back(rows);
//     taskDataPar->inputs_count.emplace_back(cols);
//     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_minima.data()));
//     taskDataPar->outputs_count.emplace_back(global_minima.size());
//   }

//   naumov_b_min_colum_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
//   ASSERT_EQ(testMpiTaskParallel.validation(), true);
//   testMpiTaskParallel.pre_processing();
//   testMpiTaskParallel.run();
//   testMpiTaskParallel.post_processing();

//   // Create Perf attributes
//   auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
//   perfAttr->num_running = 10;
//   const boost::mpi::timer current_timer;
//   perfAttr->current_timer = [&] { return current_timer.elapsed(); };

//   // Create and init perf results
//   auto perfResults = std::make_shared<ppc::core::PerfResults>();

//   // Create Perf analyzer
//   auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
//   perfAnalyzer->task_run(perfAttr, perfResults);
//   if (world.rank() == 0) {
//     ppc::core::Perf::print_perf_statistic(perfResults);
//   }
// }