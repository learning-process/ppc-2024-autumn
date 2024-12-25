#include "mpi/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_mpi.hpp"
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, SortedData) {
  boost::mpi::communicator world;
  int global = 10;
  std::vector<double> inputData = {-6.1, -5.1, 0.3, 1.0, 1.1, 2.7, 3.3, 5.4, 7.8, 9.1};
  std::vector<double> resP(global, 0.0);
  std::vector<double> resS(global, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataPar->inputs_count.emplace_back(global);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resP.data()));
    taskDataPar->outputs_count.emplace_back(global);
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel parallelTask(taskDataPar);
  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs = taskDataPar->inputs;
    taskDataSeq->inputs_count = taskDataPar->inputs_count;
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resS.data()));
    taskDataSeq->outputs_count.emplace_back(global);

    komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskSequential sequentialTask(taskDataSeq);
    ASSERT_EQ(sequentialTask.validation(), true);
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    for (int i = 0; i < global; ++i) {
      ASSERT_NEAR(resP[i], resS[i], 1e-12);
    }
  }
}


TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, AllEqualData) {
  boost::mpi::communicator world;
  int global = 10;
  std::vector<double> inputData(global, 3.14);
  std::vector<double> resP(global, 0.0);
  std::vector<double> resS(global, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataPar->inputs_count.emplace_back(global);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resP.data()));
    taskDataPar->outputs_count.emplace_back(global);
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel parallelTask(taskDataPar);
  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs = taskDataPar->inputs;
    taskDataSeq->inputs_count = taskDataPar->inputs_count;
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resS.data()));
    taskDataSeq->outputs_count.emplace_back(global);

    komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskSequential sequentialTask(taskDataSeq);
    ASSERT_EQ(sequentialTask.validation(), true);
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    for (int i = 0; i < global; ++i) {
      ASSERT_NEAR(resP[i], resS[i], 1e-12);
    }
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, EmptyData) {
  boost::mpi::communicator world;
  int global = 0;
  std::vector<double> inputData;
  std::vector<double> resP(global, 0.0);
  std::vector<double> resS(global, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataPar->inputs_count.emplace_back(global);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resP.data()));
    taskDataPar->outputs_count.emplace_back(global);
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel parallelTask(taskDataPar);
  ASSERT_EQ(parallelTask.validation(), false);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs = taskDataPar->inputs;
    taskDataSeq->inputs_count = taskDataPar->inputs_count;
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resS.data()));
    taskDataSeq->outputs_count.emplace_back(global);

    komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskSequential sequentialTask(taskDataSeq);
    ASSERT_EQ(sequentialTask.validation(), false);
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    for (int i = 0; i < global; ++i) {
      ASSERT_NEAR(resP[i], resS[i], 1e-12);
    }
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, SingleElementData) {
  boost::mpi::communicator world;
  int global = 1;
  std::vector<double> inputData = {42.0};
  std::vector<double> resP(global, 0.0);
  std::vector<double> resS(global, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataPar->inputs_count.emplace_back(global);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resP.data()));
    taskDataPar->outputs_count.emplace_back(global);
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel parallelTask(taskDataPar);
  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs = taskDataPar->inputs;
    taskDataSeq->inputs_count = taskDataPar->inputs_count;
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resS.data()));
    taskDataSeq->outputs_count.emplace_back(global);

    komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskSequential sequentialTask(taskDataSeq);
    ASSERT_EQ(sequentialTask.validation(), true);
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    for (int i = 0; i < global; ++i) {
      ASSERT_NEAR(resP[i], resS[i], 1e-12);
    }
  }
}