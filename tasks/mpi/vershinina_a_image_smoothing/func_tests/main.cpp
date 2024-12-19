
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "mpi/vershinina_a_image_smoothing/include/ops_mpi.hpp"

std::vector<int> getRandomVector(int rows, int cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distr(0, 255);
  std::vector<int> vec(rows * cols);
  int vec_size = vec.size();
  for (int i = 0; i < vec_size; i++) {
    vec[i] = distr(gen);
  }
  return vec;
}

TEST(vershinina_a_image_smoothing, Test_Example) {
  boost::mpi::communicator world;
  std::vector<int> in{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
  std::vector<int> global_res(4 * 4);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(4);
    taskDataPar->inputs_count.emplace_back(4);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }
  vershinina_a_image_smoothing::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_res(4 * 4);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(4);
    taskDataSeq->inputs_count.emplace_back(4);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    vershinina_a_image_smoothing::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_res, global_res);
  }
}

TEST(vershinina_a_image_smoothing, Test_10_max_10) {
  boost::mpi::communicator world;
  std::vector<int> in;
  std::vector<int> global_res(10 * 10);

  in = getRandomVector(10, 10);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(10);
    taskDataPar->inputs_count.emplace_back(10);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }
  vershinina_a_image_smoothing::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_res(10 * 10);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(10);
    taskDataSeq->inputs_count.emplace_back(10);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    vershinina_a_image_smoothing::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_res, global_res);
  }
}

TEST(vershinina_a_image_smoothing, Test_Random_1) {
  boost::mpi::communicator world;
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distr(2, 32);
  int random_rows = distr(gen);
  int random_cols = distr(gen);
  std::vector<int> in;
  std::vector<int> global_res(random_rows * random_cols);

  in = getRandomVector(random_rows, random_cols);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(random_rows);
    taskDataPar->inputs_count.emplace_back(random_cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }
  vershinina_a_image_smoothing::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_res(random_rows * random_cols);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(random_rows);
    taskDataSeq->inputs_count.emplace_back(random_cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    vershinina_a_image_smoothing::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_res, global_res);
  }
}
TEST(vershinina_a_image_smoothing, Test_Random_2) {
  boost::mpi::communicator world;
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distr(2, 32);
  int random_rows = distr(gen);
  int random_cols = distr(gen);
  std::vector<int> in;
  std::vector<int> global_res(random_rows * random_cols);

  in = getRandomVector(random_rows, random_cols);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(random_rows);
    taskDataPar->inputs_count.emplace_back(random_cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }
  vershinina_a_image_smoothing::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_res(random_rows * random_cols);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(random_rows);
    taskDataSeq->inputs_count.emplace_back(random_cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    vershinina_a_image_smoothing::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_res, global_res);
  }
}