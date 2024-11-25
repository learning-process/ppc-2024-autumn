#include <gtest/gtest.h>

#include <boost/mpi/environment.hpp>
#include "mpi/kudryashova_i_gather my/include/GatherMyMPI.hpp"

static int seedOffset = 0;
std::vector<int> GetRandomVectorGatherMy(int size) {
  std::vector<int> vector(size);
  std::srand(static_cast<unsigned>(time(nullptr)) + ++seedOffset);
  for (int i = 0; i < size; ++i) {
    vector[i] = static_cast<int>(std::rand() % 100 + 1.0f);
  }
  return vector;
}

TEST(kudryashova_i_gather_my_mpi, mpi_vector_dot_product_func) {
  std::vector<int> vector1 = {8, 7, 6};
  std::vector<int> vector2 = {3, 2, 1};
  ASSERT_EQ(44, kudryashova_i_gather_my::vectorDotProductGather(vector1, vector2));
}

TEST(kudryashova_i_gather_my_mpi, dot_product_vector_120) {
  boost::mpi::communicator world;
  const int count_size_vector = 120;
  std::vector<uint8_t> global_vector;
  std::vector<int32_t> result(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::vector<int> vector1 = GetRandomVectorGatherMy(count_size_vector);
  std::vector<int> vector2 = GetRandomVectorGatherMy(count_size_vector);

  if (world.rank() == 0) {
    global_vector.reserve(vector1.size() + vector2.size());
    global_vector.insert(global_vector.end(), vector1.begin(), vector1.end());
    global_vector.insert(global_vector.end(), vector2.begin(), vector2.end());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }
  auto testMpiTaskParallel = kudryashova_i_gather_my::TestMPITaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int32_t> reference(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataSeq->inputs_count.emplace_back(global_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());
    kudryashova_i_gather_my::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference[0], result[0]);
    ASSERT_EQ(kudryashova_i_gather_my::vectorDotProductGather(vector1, vector2), result[0]);

  }
}

TEST(kudryashova_i_gather_my_mpi, dot_product_vector_360) {
  boost::mpi::communicator world;
  const int count_size_vector = 360;
  std::vector<uint8_t> global_vector;
  std::vector<int32_t> result(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::vector<int> vector1 = GetRandomVectorGatherMy(count_size_vector);
  std::vector<int> vector2 = GetRandomVectorGatherMy(count_size_vector);
  if (world.rank() == 0) {
    global_vector.reserve(vector1.size() + vector2.size());
    global_vector.insert(global_vector.end(), vector1.begin(), vector1.end());
    global_vector.insert(global_vector.end(), vector2.begin(), vector2.end());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }
  auto testMpiTaskParallel = kudryashova_i_gather_my::TestMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int32_t> reference(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataSeq->inputs_count.emplace_back(global_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());
    kudryashova_i_gather_my::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference[0], result[0]);
  }
}

TEST(kudryashova_i_gather_my_mpi, check_vectors_equal) {
  boost::mpi::communicator world;
  const int count_size_vector = 100;
  std::vector<int> global_vector;
  std::vector<int32_t> result(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::vector<int> vector1 = GetRandomVectorGatherMy(count_size_vector);
  std::vector<int> vector2 = GetRandomVectorGatherMy(count_size_vector);
  if (world.rank() == 0) {
    global_vector.reserve(vector1.size() + vector2.size());
    global_vector.insert(global_vector.end(), vector1.begin(), vector1.end());
    global_vector.insert(global_vector.end(), vector2.begin(), vector2.end());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }
  kudryashova_i_gather_my::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
}

TEST(kudryashova_i_gather_my_mpi, check_not_equal_vectors) {
  boost::mpi::communicator world;
  const int count_size_vector = 100;
  std::vector<int> global_vector;
  std::vector<int32_t> result(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::vector<int> vector1 = GetRandomVectorGatherMy(count_size_vector);
  std::vector<int> vector2 = GetRandomVectorGatherMy(count_size_vector + 1);
  if (world.rank() == 0) {
    global_vector.reserve(vector1.size() + vector2.size());
    global_vector.insert(global_vector.end(), vector1.begin(), vector1.end());
    global_vector.insert(global_vector.end(), vector2.begin(), vector2.end());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }
  kudryashova_i_gather_my::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
}

TEST(kudryashova_i_gather_my_mpi, check_dot_product_empty_vectors) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vector;
  std::vector<int32_t> result(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<int> vector1 = {};
    std::vector<int> vector2 = {};
    global_vector.reserve(vector1.size() + vector2.size());
    global_vector.insert(global_vector.end(), vector1.begin(), vector1.end());
    global_vector.insert(global_vector.end(), vector2.begin(), vector2.end());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
    kudryashova_i_gather_my::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(kudryashova_i_gather_my_mpi, dot_product_vector_1) {
  boost::mpi::communicator world;
  const int count_size_vector = 1;
  std::vector<int> global_vector;
  std::vector<int32_t> result(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::vector<int> vector1 = GetRandomVectorGatherMy(count_size_vector);
  std::vector<int> vector2 = GetRandomVectorGatherMy(count_size_vector);
  std::vector<int> size_of_vectors;
  size_of_vectors.push_back(vector1.size());
  size_of_vectors.push_back(vector2.size());
  if (world.rank() == 0) {
    global_vector.reserve(vector1.size() + vector2.size());
    global_vector.insert(global_vector.end(), vector1.begin(), vector1.end());
    global_vector.insert(global_vector.end(), vector2.begin(), vector2.end());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }
  auto testMpiTaskParallel = kudryashova_i_gather_my::TestMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int32_t> reference(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataSeq->inputs_count.emplace_back(global_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());
    kudryashova_i_gather_my::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference[0], result[0]);
  }
}

TEST(kudryashova_i_gather_my_mpi, check_dot_product_empty_and_nonempty_vectors) {
  boost::mpi::communicator world;
  std::vector<int> global_vector;
  std::vector<int32_t> result(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<int> vector1 = {};
    std::vector<int> vector2 = {10};
    global_vector.reserve(vector1.size() + vector2.size());
    global_vector.insert(global_vector.end(), vector1.begin(), vector1.end());
    global_vector.insert(global_vector.end(), vector2.begin(), vector2.end());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());

    kudryashova_i_gather_my::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(kudryashova_i_gather_my_mpi, dot_product_vector_1_with_zero) {
  boost::mpi::communicator world;
  std::vector<int> global_vector;
  std::vector<int32_t> result(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<int> vector1 = {0};
    std::vector<int> vector2 = {1};
    global_vector.reserve(vector1.size() + vector2.size());
    global_vector.insert(global_vector.end(), vector1.begin(), vector1.end());
    global_vector.insert(global_vector.end(), vector2.begin(), vector2.end());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());

    kudryashova_i_gather_my::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
  }
}