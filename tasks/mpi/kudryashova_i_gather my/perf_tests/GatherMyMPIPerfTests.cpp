#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/kudryashova_i_gather my/include/GatherMyMPI.hpp"

static int seedOffset = 0;
std::vector<int> GetRandomVectorForGatherMy(int size) {
  std::vector<int> vector(size);
  std::srand(static_cast<unsigned>(time(nullptr)) + ++seedOffset);
  for (int i = 0; i < size; ++i) {
    vector[i] = std::rand() % 100 + 1;
  }
  return vector;
}

TEST(kudryashova_i_gather_my, test_pipeline_run) {
  boost::mpi::communicator world;
  const int count = 15000000;
  std::vector<uint8_t> global_vector;
  std::vector<int> vector1 = GetRandomVectorForGatherMy(count);
  std::vector<int> vector2 = GetRandomVectorForGatherMy(count);
  std::vector<int32_t> result(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vector.resize(vector1.size() + vector2.size());
    std::copy(vector1.begin(), vector1.end(), global_vector.begin());
    std::copy(vector2.begin(), vector2.end(), global_vector.begin() + vector1.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }
  auto testMpiTaskParallel = std::make_shared<kudryashova_i_gather_my::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(kudryashova_i_gather_my::vectorDotProductGather(vector1, vector2), result[0]);
  }
}

TEST(kudryashova_i_gather_my, test_task_run) {
  boost::mpi::communicator world;
  const int count_size_vector = 15000000;
  std::vector<uint8_t> global_vector;
  std::vector<int> vector1 = GetRandomVectorForGatherMy(count_size_vector);
  std::vector<int> vector2 = GetRandomVectorForGatherMy(count_size_vector);
  std::vector<int32_t> result(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vector.resize(vector1.size() + vector2.size());
    std::copy(vector1.begin(), vector1.end(), global_vector.begin());
    std::copy(vector2.begin(), vector2.end(), global_vector.begin() + vector1.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }
  auto testMpiTaskParallel = std::make_shared<kudryashova_i_gather_my::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(kudryashova_i_gather_my::vectorDotProductGather(vector1, vector2), result[0]);
  }
}