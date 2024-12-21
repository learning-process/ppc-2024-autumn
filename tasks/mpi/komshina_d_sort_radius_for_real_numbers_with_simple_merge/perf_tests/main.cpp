#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_mpi.hpp"

namespace mpi = boost::mpi;

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, test_pipeline_run) {
  mpi::environment env;
  mpi::communicator world;

  std::vector<double> inputData = {5.4, -3.1, 7.2, 0.0, -8.5, 2.3, -1.1, 4.4};
  std::vector<double> xPar(inputData.size(), 0.0);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataPar->inputs_count.emplace_back(inputData.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(xPar.data()));
    taskDataPar->outputs_count.emplace_back(xPar.size());
  }

  auto parallelRadixSort =
      std::make_shared<komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel>(taskDataPar);

  ASSERT_TRUE(parallelRadixSort->validation()) << "Validation failed!";
  parallelRadixSort->pre_processing();
  parallelRadixSort->run();
  parallelRadixSort->post_processing();

  if (world.rank() == 0) {
    std::vector<double> expectedOutput = {-8.5, -3.1, -1.1, 0.0, 2.3, 4.4, 5.4, 7.2};
    ASSERT_EQ(xPar, expectedOutput) << "Output data does not match expected!";
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, test_task_run) {
  mpi::environment env;
  mpi::communicator world;

  std::vector<double> inputData = {10.5, -12.3, 0.0, 8.4, -2.2, 3.3, 1.1, -5.5};
  std::vector<double> xPar(inputData.size(), 0.0);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataPar->inputs_count.emplace_back(inputData.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(xPar.data()));
    taskDataPar->outputs_count.emplace_back(xPar.size());
  }

  auto parallelRadixSort =
      std::make_shared<komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel>(taskDataPar);

  ASSERT_TRUE(parallelRadixSort->validation()) << "Validation failed!";
  parallelRadixSort->pre_processing();
  parallelRadixSort->run();
  parallelRadixSort->post_processing();

  if (world.rank() == 0) {
    std::vector<double> expectedOutput = {-12.3, -5.5, -2.2, 0.0, 1.1, 3.3, 8.4, 10.5};
    ASSERT_EQ(xPar, expectedOutput) << "Output data does not match expected!";
  }
}