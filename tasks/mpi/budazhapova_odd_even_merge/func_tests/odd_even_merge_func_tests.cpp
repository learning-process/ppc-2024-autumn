#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "mpi/budazhapova_odd_even_merge/include/odd_even_merge.hpp"

namespace budazhapova_betcher_odd_even_merge_mpi {
std::vector<int> generateRandomVector(int size, int minValue, int maxValue) {
  std::vector<int> randomVector;
  randomVector.reserve(size);
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (int i = 0; i < size; ++i) {
    int randomNum = std::rand() % (maxValue - minValue + 1) + minValue;
    randomVector.push_back(randomNum);
  }
  return randomVector;
}
}  // namespace budazhapova_betcher_odd_even_merge_mpi

TEST(budazhapova_betcher_odd_even_merge_mpi, ordinary_test) {
  boost::mpi::communicator world;
  std::vector<int> input_vector = {34, 12, 5, 78, 23, 45, 67, 89, 10, 2, 56, 43, 91, 15, 30};
  std::vector<int> out(15, 0);
  std::vector<int> out_seq(15, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    taskDataPar->inputs_count.emplace_back(input_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  budazhapova_betcher_odd_even_merge_mpi::MergeParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    taskDataSeq->inputs_count.emplace_back(input_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    budazhapova_betcher_odd_even_merge_mpi::MergeSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(out, out_seq);
  }
}

TEST(budazhapova_betcher_odd_even_merge_mpi, random_vector_test) {
  boost::mpi::communicator world;
  std::vector<int> input_vector = budazhapova_betcher_odd_even_merge_mpi::generateRandomVector(100, 5, 100);
  std::vector<int> out(100, 0);
  std::vector<int> out_seq(100, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    taskDataPar->inputs_count.emplace_back(input_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  budazhapova_betcher_odd_even_merge_mpi::MergeParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    taskDataSeq->inputs_count.emplace_back(input_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    budazhapova_betcher_odd_even_merge_mpi::MergeSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(out, out_seq);
  }
}

TEST(budazhapova_betcher_odd_even_merge_mpi, validation_test) {
  boost::mpi::communicator world;
  std::vector<int> input_vector = {};
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    taskDataPar->inputs_count.emplace_back(input_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
    budazhapova_betcher_odd_even_merge_mpi::MergeParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}
