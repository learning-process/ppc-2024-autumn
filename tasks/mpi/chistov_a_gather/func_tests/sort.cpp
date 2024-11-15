#include "mpi/chistov_a_gather/include/sort.hpp"

#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

TEST(chistov_a_gather, reference_empty_vector_check) {
  boost::mpi::communicator world;
  const int count_size_vector = 100;
  std::vector<int> vector;
  std::vector<int> gathered_vec(count_size_vector);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
    taskDataPar->inputs_count.emplace_back(vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_vec.data()));
    taskDataPar->outputs_count.emplace_back(gathered_vec.size());
    chistov_a_gather::Reference<int> testMpiTaskParallel(taskDataPar);

    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(chistov_a_gather, my_empty_vector_check) {
  boost::mpi::communicator world;
  const int count_size_vector = 100;
  std::vector<int> vector;
  std::vector<int> gathered_vec(count_size_vector);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
    taskDataPar->inputs_count.emplace_back(vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_vec.data()));
    taskDataPar->outputs_count.emplace_back(gathered_vec.size());
    chistov_a_gather::Sorting<int> testMpiTaskParallel(taskDataPar);

    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(chistov_a_gather, reference_empty_output_check) {
  boost::mpi::communicator world;
  const int count_size_vector = 100;
  std::vector<int> vector;
  std::vector<int> gathered_vec;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    vector = chistov_a_gather::getRandomVector<int>(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
    taskDataPar->inputs_count.emplace_back(vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_vec.data()));
    taskDataPar->outputs_count.emplace_back(gathered_vec.size());
    chistov_a_gather::Reference<int> testMpiTaskParallel(taskDataPar);

    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(chistov_a_gather, my_empty_output_check) {
  boost::mpi::communicator world;
  const int count_size_vector = 100;
  std::vector<int> vector;
  std::vector<int> gathered_vec;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    vector = chistov_a_gather::getRandomVector<int>(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
    taskDataPar->inputs_count.emplace_back(vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_vec.data()));
    taskDataPar->outputs_count.emplace_back(gathered_vec.size());
    chistov_a_gather::Sorting<int> testMpiTaskParallel(taskDataPar);

    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(chistov_a_gather, reference_task_check) {
  boost::mpi::communicator world;
  const int count_size_vector = 10;
  std::vector<int> local_vector;
  std::vector<int> gathered_vector(count_size_vector * world.size());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  local_vector = chistov_a_gather::getRandomVector<int>(count_size_vector);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(local_vector.data()));
  taskDataPar->inputs_count.emplace_back(count_size_vector);

  if (world.rank() == 0) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_vector.data()));
    taskDataPar->outputs_count.emplace_back(gathered_vector.size());
  }

  chistov_a_gather::Reference<int> testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_TRUE(std::is_sorted(gathered_vector.begin(), gathered_vector.end()));
  }
}

TEST(chistov_a_gather, my_task_check_int) {
  boost::mpi::communicator world;
  const int count_size_vector = 10;
  std::vector<int> local_vector;
  std::vector<int> gathered_vector(count_size_vector * world.size());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  local_vector = chistov_a_gather::getRandomVector<int>(count_size_vector);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(local_vector.data()));
  taskDataPar->inputs_count.emplace_back(count_size_vector);

  if (world.rank() == 0) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_vector.data()));
    taskDataPar->outputs_count.emplace_back(gathered_vector.size());
  }

  chistov_a_gather::Sorting<int> testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_TRUE(std::is_sorted(gathered_vector.begin(), gathered_vector.end()));
  }
}

TEST(chistov_a_gather, my_task_check_double) {
  boost::mpi::communicator world;
  const int count_size_vector = 10;
  std::vector<double> local_vector;
  std::vector<double> gathered_vector(count_size_vector * world.size());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  local_vector = chistov_a_gather::getRandomVector<double>(count_size_vector);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(local_vector.data()));
  taskDataPar->inputs_count.emplace_back(count_size_vector);

  if (world.rank() == 0) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_vector.data()));
    taskDataPar->outputs_count.emplace_back(gathered_vector.size());
  }

  chistov_a_gather::Sorting<double> testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_TRUE(std::is_sorted(gathered_vector.begin(), gathered_vector.end()));
  }
}

TEST(chistov_a_gather, my_task_check_float) {
  boost::mpi::communicator world;
  const int count_size_vector = 10;
  std::vector<float> local_vector;
  std::vector<float> gathered_vector(count_size_vector * world.size());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  local_vector = chistov_a_gather::getRandomVector<float>(count_size_vector);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(local_vector.data()));
  taskDataPar->inputs_count.emplace_back(count_size_vector);

  if (world.rank() == 0) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_vector.data()));
    taskDataPar->outputs_count.emplace_back(gathered_vector.size());
  }

  chistov_a_gather::Sorting<float> testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_TRUE(std::is_sorted(gathered_vector.begin(), gathered_vector.end()));
  }
}

TEST(chistov_a_gather, comprasion_of_two_implementations) {
  boost::mpi::communicator world;
  const int count_size_vector = 10;
  std::vector<int> local_vector;
  std::vector<int> gathered_vector1(count_size_vector * world.size());
  std::vector<int> gathered_vector2(count_size_vector * world.size());

  std::shared_ptr<ppc::core::TaskData> taskDataPar1 = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataPar2 = std::make_shared<ppc::core::TaskData>();

  local_vector = chistov_a_gather::getRandomVector<int>(count_size_vector);
  taskDataPar1->inputs.emplace_back(reinterpret_cast<uint8_t*>(local_vector.data()));
  taskDataPar1->inputs_count.emplace_back(count_size_vector);
  taskDataPar2->inputs.emplace_back(reinterpret_cast<uint8_t*>(local_vector.data()));
  taskDataPar2->inputs_count.emplace_back(count_size_vector);

  if (world.rank() == 0) {
    taskDataPar1->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_vector1.data()));
    taskDataPar1->outputs_count.emplace_back(gathered_vector1.size());
    taskDataPar2->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_vector2.data()));
    taskDataPar2->outputs_count.emplace_back(gathered_vector2.size());
  }

  chistov_a_gather::Sorting<int> testMpiTaskParallel1(taskDataPar1);
  ASSERT_EQ(testMpiTaskParallel1.validation(), true);
  testMpiTaskParallel1.pre_processing();
  testMpiTaskParallel1.run();
  testMpiTaskParallel1.post_processing();

  chistov_a_gather::Reference<int> testMpiTaskParallel2(taskDataPar2);
  ASSERT_EQ(testMpiTaskParallel2.validation(), true);
  testMpiTaskParallel2.pre_processing();
  testMpiTaskParallel2.run();
  testMpiTaskParallel2.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(gathered_vector1, gathered_vector2);
  }
}