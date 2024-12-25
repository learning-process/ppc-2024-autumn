#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "mpi/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_mpi.hpp"

namespace mpi = boost::mpi;

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, SimpleData) {
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int count = 8;
  std::vector<double> inputData = {10.1, 8.1, 0.2, 1.5, -6.3, 4.4, -11.4, 0.6};
  std::vector<double> resPar(count, 0.0);
  std::vector<double> resSeq(count, 0.0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataPar->inputs_count.emplace_back(count);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resPar.data()));
    taskDataPar->outputs_count.emplace_back(count);

    taskDataSeq->inputs = taskDataPar->inputs;
    taskDataSeq->inputs_count = taskDataPar->inputs_count;

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(count);
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel sortTaskPar(taskDataPar);
  ASSERT_TRUE(sortTaskPar.validation());
  sortTaskPar.pre_processing();
  sortTaskPar.run();
  sortTaskPar.post_processing();

  if (world.rank() == 0) {
    komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskSequential sortTaskSeq(taskDataSeq);
    ASSERT_TRUE(sortTaskSeq.validation());
    sortTaskSeq.pre_processing();
    sortTaskSeq.run();
    sortTaskSeq.post_processing();

    auto* resultPar = reinterpret_cast<double*>(taskDataPar->outputs[0]);
    auto* resultSeq = reinterpret_cast<double*>(taskDataSeq->outputs[0]);

    for (int i = 0; i < count; ++i) {
      ASSERT_NEAR(resultPar[i], resultSeq[i], 1e-12);
    }
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, EmptyData) {
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int count = 0;
  std::vector<double> inputData;
  std::vector<double> resPar(count, 0.0);
  std::vector<double> resSeq(count, 0.0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataPar->inputs_count.emplace_back(count);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resPar.data()));
    taskDataPar->outputs_count.emplace_back(count);

    taskDataSeq->inputs = taskDataPar->inputs;
    taskDataSeq->inputs_count = taskDataPar->inputs_count;

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(count);
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel sortTaskPar(taskDataPar);
  ASSERT_TRUE(sortTaskPar.validation());
  sortTaskPar.pre_processing();
  sortTaskPar.run();
  sortTaskPar.post_processing();

  if (world.rank() == 0) {
    komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskSequential sortTaskSeq(taskDataSeq);
    ASSERT_TRUE(sortTaskSeq.validation());
    sortTaskSeq.pre_processing();
    sortTaskSeq.run();
    sortTaskSeq.post_processing();

    auto* resultPar = reinterpret_cast<double*>(taskDataPar->outputs[0]);
    auto* resultSeq = reinterpret_cast<double*>(taskDataSeq->outputs[0]);

    for (int i = 0; i < count; ++i) {
      ASSERT_NEAR(resultPar[i], resultSeq[i], 1e-12);
    }
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, SingleElementData) {
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int count = 1;
  std::vector<double> inputData = {42.0};
  std::vector<double> resPar(count, 0.0);
  std::vector<double> resSeq(count, 0.0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataPar->inputs_count.emplace_back(count);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resPar.data()));
    taskDataPar->outputs_count.emplace_back(count);

    taskDataSeq->inputs = taskDataPar->inputs;
    taskDataSeq->inputs_count = taskDataPar->inputs_count;

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(count);
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel sortTaskPar(taskDataPar);
  ASSERT_TRUE(sortTaskPar.validation());
  sortTaskPar.pre_processing();
  sortTaskPar.run();
  sortTaskPar.post_processing();

  if (world.rank() == 0) {
    komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskSequential sortTaskSeq(taskDataSeq);
    ASSERT_TRUE(sortTaskSeq.validation());
    sortTaskSeq.pre_processing();
    sortTaskSeq.run();
    sortTaskSeq.post_processing();

    auto* resultPar = reinterpret_cast<double*>(taskDataPar->outputs[0]);
    auto* resultSeq = reinterpret_cast<double*>(taskDataSeq->outputs[0]);

    ASSERT_NEAR(resultPar[0], resultSeq[0], 1e-12);
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, LargeData) {
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int count = 100000;
  std::vector<double> inputData(count);
  std::generate(inputData.begin(), inputData.end(), std::rand);
  std::vector<double> resPar(count, 0.0);
  std::vector<double> resSeq(count, 0.0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataPar->inputs_count.emplace_back(count);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resPar.data()));
    taskDataPar->outputs_count.emplace_back(count);

    taskDataSeq->inputs = taskDataPar->inputs;
    taskDataSeq->inputs_count = taskDataPar->inputs_count;

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(count);
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel sortTaskPar(taskDataPar);
  ASSERT_TRUE(sortTaskPar.validation());
  sortTaskPar.pre_processing();
  sortTaskPar.run();
  sortTaskPar.post_processing();

  if (world.rank() == 0) {
    komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskSequential sortTaskSeq(taskDataSeq);
    ASSERT_TRUE(sortTaskSeq.validation());
    sortTaskSeq.pre_processing();
    sortTaskSeq.run();
    sortTaskSeq.post_processing();

    auto* resultPar = reinterpret_cast<double*>(taskDataPar->outputs[0]);
    auto* resultSeq = reinterpret_cast<double*>(taskDataSeq->outputs[0]);

    for (int i = 0; i < count; ++i) {
      ASSERT_NEAR(resultPar[i], resultSeq[i], 1e-12);
    }
  }
}