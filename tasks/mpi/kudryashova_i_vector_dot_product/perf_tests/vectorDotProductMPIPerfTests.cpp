#include <gtest/gtest.h>
#include <boost/mpi/timer.hpp>
#include <vector>
#include "core/perf/include/perf.hpp"
#include "mpi/kudryashova_i_vector_dot_product/include/vectorDotProductMPI.hpp"

//TEST(kudryashova_i_vector_dot_product_mpi, test_pipeline_run) {
//  const int count = 15000000;
//  boost::mpi::communicator world;
//  std::vector<std::vector<int>> global_vector;
//  std::vector<int> vector1 = kudryashova_i_vector_dot_product_mpi::getRandomVector(count);
//  std::vector<int> vector2 = kudryashova_i_vector_dot_product_mpi::getRandomVector(count);
//  std::vector<int32_t> result(1, 0);
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//  if (world.rank() == 0) {
//    global_vector = {vector1, vector2};
//    for (size_t i = 0; i < global_vector.size(); i++) {
//      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector[i].data()));
//    }
//    taskDataPar->inputs_count.emplace_back(global_vector[0].size());
//    taskDataPar->inputs_count.emplace_back(global_vector[1].size());
//     
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
//    taskDataPar->outputs_count.emplace_back(result.size());
//  }
//  auto testMpiTaskParallel = std::make_shared<kudryashova_i_vector_dot_product_mpi::TestMPITaskParallel>(taskDataPar);
//  ASSERT_EQ(testMpiTaskParallel->validation(), true);
//  testMpiTaskParallel->pre_processing();
//  testMpiTaskParallel->run();
//  testMpiTaskParallel->post_processing();
//  
//  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
//  perfAttr->num_running = 10;
//  const boost::mpi::timer current_timer;
//  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
//  
//  auto perfResults = std::make_shared<ppc::core::PerfResults>();
//  
//  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
//  perfAnalyzer->pipeline_run(perfAttr, perfResults);
//  if (world.rank() == 0) {
//    ppc::core::Perf::print_perf_statistic(perfResults);
//    ASSERT_EQ(kudryashova_i_vector_dot_product_mpi::vectorDotProduct(global_vector[0], global_vector[1]), result[0]);
//  }
//}
//
//TEST(kudryashova_i_vector_dot_product_mpi, test_task_run) {
//  const int count_size_vector = 15000000;
//  boost::mpi::communicator world;
//  std::vector<std::vector<int>> global_vector;
//  std::vector<int> vector1 = kudryashova_i_vector_dot_product_mpi::getRandomVector(count_size_vector);
//  std::vector<int> vector2 = kudryashova_i_vector_dot_product_mpi::getRandomVector(count_size_vector);
//  std::vector<int32_t> result(1, 0);
//  // Create TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//  if (world.rank() == 0) {
//    global_vector = {vector1, vector2};
//    for (size_t i = 0; i < global_vector.size(); i++) {
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector[i].data()));
//    }
//    taskDataPar->inputs_count.emplace_back(global_vector[0].size());
//    taskDataPar->inputs_count.emplace_back(global_vector[1].size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
//    taskDataPar->outputs_count.emplace_back(result.size());
//  }
//  
//  auto testMpiTaskParallel = std::make_shared<kudryashova_i_vector_dot_product_mpi::TestMPITaskParallel>(taskDataPar);
//  ASSERT_EQ(testMpiTaskParallel->validation(), true);
//  testMpiTaskParallel->pre_processing();
//  testMpiTaskParallel->run();
//  testMpiTaskParallel->post_processing();
//  
//  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
//  perfAttr->num_running = 10;
//  const boost::mpi::timer current_timer;
//  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
//  auto perfResults = std::make_shared<ppc::core::PerfResults>();
//  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
//  perfAnalyzer->task_run(perfAttr, perfResults);
//  if (world.rank() == 0) {
//    ppc::core::Perf::print_perf_statistic(perfResults);
//    ASSERT_EQ(kudryashova_i_vector_dot_product_mpi::vectorDotProduct(global_vector[0], global_vector[1]), result[0]);
//  }   
//}