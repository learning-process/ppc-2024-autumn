#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/komshina_d_min_of_vector_elements/include/ops_mpi.hpp"

TEST(komshina_d_min_of_vector_elements_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  const int count = 5'000'000;
  const int start = 500;
  const int min = -10;

  std::vector<int> in(count, start);
  std::vector<int32_t> global_min(1, 0);  

  std::random_device dev;
  std::mt19937 gen(dev());

  for (int i = 0; i < count - 1; i++) {
    in[i] = gen() % 1000; 
  }
  in[count - 10] = min;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(min, global_min[0]);  
  }
}

TEST(komshina_d_min_of_vector_elements_mpi, test_task_run) {
  boost::mpi::communicator world;
  const int count = 5'000'000;
  const int start = 500;
  const int min = -10;

  std::vector<int> in(count, start);
  std::vector<int32_t> global_min(1, 0);  


  std::random_device dev;
  std::mt19937 gen(dev());

 
  for (int i = 0; i < count - 1; i++) {
    in[i] = gen() % 1000;  
  }
  
  in[count - 10] = min;

  
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(min, global_min[0]);  
  }
}
