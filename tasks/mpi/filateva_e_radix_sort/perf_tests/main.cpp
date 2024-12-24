// Filateva Elizaveta Radix Sort
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/filateva_e_radix_sort/include/ops_mpi.hpp"

namespace filateva_e_radix_sort_mpi {

void GeneratorVector(std::vector<int> &vec) {
  int max_z = 100000;
  int min_z = -100000;
  std::random_device dev;
  std::mt19937 gen(dev());
  for (unsigned long i = 0; i < vec.size(); i++) {
    vec[i] = gen() % (max_z - min_z + 1) + min_z;
  }
}

}  // namespace filateva_e_radix_sort_mpi

TEST(filateva_e_radix_sort_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int size = 400000;
  std::vector<int> vec;
  std::vector<int> answer;
  std::vector<int> tResh;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    vec.resize(size);
    answer.resize(size);

    filateva_e_radix_sort_mpi::GeneratorVector(vec);
    tResh = vec;

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  auto radixSort = std::make_shared<filateva_e_radix_sort_mpi::RadixSort>(taskData);
  ASSERT_TRUE(radixSort->validation());
  radixSort->pre_processing();
  radixSort->run();
  radixSort->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(radixSort);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    std::sort(tResh.begin(), tResh.end());

    EXPECT_EQ(answer.size(), tResh.size());
    for (int i = 0; i < size; i++) {
      EXPECT_EQ(answer[i], tResh[i]);
    }
  }
}

TEST(filateva_e_radix_sort_mpi, test_task_run) {
  boost::mpi::communicator world;
  int size = 400000;
  std::vector<int> vec;
  std::vector<int> answer;
  std::vector<int> tResh;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    vec.resize(size);
    answer.resize(size);

    filateva_e_radix_sort_mpi::GeneratorVector(vec);
    tResh = vec;

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  auto radixSort = std::make_shared<filateva_e_radix_sort_mpi::RadixSort>(taskData);
  ASSERT_TRUE(radixSort->validation());
  radixSort->pre_processing();
  radixSort->run();
  radixSort->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(radixSort);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    std::sort(tResh.begin(), tResh.end());

    EXPECT_EQ(answer.size(), tResh.size());
    for (int i = 0; i < size; i++) {
      EXPECT_EQ(answer[i], tResh[i]);
    }
  }
}