#include "mpi/chistov_a_gather_my/include/sort_my.hpp"

#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

TEST(chistov_a_gather_my, my_empty_vector_check) {
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
    chistov_a_gather_my::Sorting<int> testMpiTaskParallel(taskDataPar);

    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(chistov_a_gather_my, my_empty_output_check) {
  boost::mpi::communicator world;
  const int count_size_vector = 100;
  std::vector<int> vector;
  std::vector<int> gathered_vec;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    vector = chistov_a_gather_my::getRandomVector<int>(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
    taskDataPar->inputs_count.emplace_back(vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_vec.data()));
    taskDataPar->outputs_count.emplace_back(gathered_vec.size());
    chistov_a_gather_my::Sorting<int> testMpiTaskParallel(taskDataPar);

    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(chistov_a_gather_my, my_task_check_int) {
  boost::mpi::communicator world;
  const int count_size_vector = 10;
  std::vector<int> local_vector;
  std::vector<int> vector;
  std::vector<int> gathered_vector(count_size_vector * world.size());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  local_vector = chistov_a_gather_my::getRandomVector<int>(count_size_vector);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(local_vector.data()));
  taskDataPar->inputs_count.emplace_back(count_size_vector);

  if (world.rank() == 0) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_vector.data()));
    taskDataPar->outputs_count.emplace_back(gathered_vector.size());
  }

  chistov_a_gather_my::Sorting<int> testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  boost::mpi::gather(world, local_vector.data(), count_size_vector, vector, 0);
  if (world.rank() == 0) {
    std::sort(vector.begin(), vector.end());
    ASSERT_EQ(gathered_vector, vector);
  }
}

TEST(chistov_a_gather_my, my_task_check_double) {
  boost::mpi::communicator world;
  const int count_size_vector = 10;
  std::vector<double> local_vector;
  std::vector<double> vector;
  std::vector<double> gathered_vector(count_size_vector * world.size());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  local_vector = chistov_a_gather_my::getRandomVector<double>(count_size_vector);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(local_vector.data()));
  taskDataPar->inputs_count.emplace_back(count_size_vector);

  if (world.rank() == 0) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_vector.data()));
    taskDataPar->outputs_count.emplace_back(gathered_vector.size());
  }

  chistov_a_gather_my::Sorting<double> testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  boost::mpi::gather(world, local_vector.data(), count_size_vector, vector, 0);
  if (world.rank() == 0) {
    std::sort(vector.begin(), vector.end());
    ASSERT_EQ(gathered_vector, vector);
  }
}

TEST(chistov_a_gather_my, my_task_check_float) {
  boost::mpi::communicator world;
  const int count_size_vector = 10;
  std::vector<float> local_vector;
  std::vector<float> vector;
  std::vector<float> gathered_vector(count_size_vector * world.size());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  local_vector = chistov_a_gather_my::getRandomVector<float>(count_size_vector);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(local_vector.data()));
  taskDataPar->inputs_count.emplace_back(count_size_vector);

  if (world.rank() == 0) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_vector.data()));
    taskDataPar->outputs_count.emplace_back(gathered_vector.size());
  }

  chistov_a_gather_my::Sorting<float> testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  boost::mpi::gather(world, local_vector.data(), count_size_vector, vector, 0);
  if (world.rank() == 0) {
    std::sort(vector.begin(), vector.end());
    ASSERT_EQ(gathered_vector, vector);
  }
}

TEST(chistov_a_gather_my, boost_task_check_char) {
  boost::mpi::communicator world;
  const int count_size_vector = 10;
  std::vector<char> local_vector;
  std::vector<char> vector;
  std::vector<char> gathered_vector(count_size_vector * world.size());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  local_vector = chistov_a_gather_my::getRandomVector<char>(count_size_vector);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(local_vector.data()));
  taskDataPar->inputs_count.emplace_back(count_size_vector);

  if (world.rank() == 0) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_vector.data()));
    taskDataPar->outputs_count.emplace_back(gathered_vector.size());
  }

  chistov_a_gather_my::Sorting<char> testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  boost::mpi::gather(world, local_vector.data(), count_size_vector, vector, 0);
  if (world.rank() == 0) {
    std::sort(vector.begin(), vector.end());
    ASSERT_EQ(gathered_vector, vector);
  }
}

TEST(chistov_a_gather_my, test_sort_fixed_values) {
  boost::mpi::communicator world;
  const int count_size_vector = 2;
  std::vector<int> local_vector;
  std::vector<int> vector;
  std::vector<int> gathered_vector(count_size_vector * world.size());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    local_vector = {1, 2};
  } else {
    local_vector = {3, 4};
  }

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(local_vector.data()));
  taskDataPar->inputs_count.emplace_back(count_size_vector);

  if (world.rank() == 0) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_vector.data()));
    taskDataPar->outputs_count.emplace_back(gathered_vector.size());
  }

  chistov_a_gather_my::Sorting<int> testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  boost::mpi::gather(world, local_vector.data(), count_size_vector, vector, 0);
  if (world.rank() == 0) {
    std::sort(vector.begin(), vector.end());
    ASSERT_EQ(gathered_vector, vector);
  }
}

TEST(chistov_a_gather_my, test_sort_different_values) {
  boost::mpi::communicator world;
  const int count_size_vector = 2;
  std::vector<int> local_vector = {world.rank(), world.rank() + 1, world.rank() + 2};
  std::vector<int> vector;
  std::vector<int> gathered_vector(count_size_vector * world.size());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(local_vector.data()));
  taskDataPar->inputs_count.emplace_back(count_size_vector);

  if (world.rank() == 0) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_vector.data()));
    taskDataPar->outputs_count.emplace_back(gathered_vector.size());
  }

  chistov_a_gather_my::Sorting<int> testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  boost::mpi::gather(world, local_vector.data(), count_size_vector, vector, 0);
  if (world.rank() == 0) {
    std::sort(vector.begin(), vector.end());
    ASSERT_EQ(gathered_vector, vector);
  }
}

TEST(chistov_a_gather_my, test_count_is_a_powers_of_two_sort) {
  boost::mpi::communicator world;
  const int count_size_vector = 32;
  std::vector<int> local_vector;
  std::vector<int> vector;
  std::vector<int> gathered_vector(count_size_vector * world.size());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  local_vector = chistov_a_gather_my::getRandomVector<int>(count_size_vector);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(local_vector.data()));
  taskDataPar->inputs_count.emplace_back(count_size_vector);

  if (world.rank() == 0) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_vector.data()));
    taskDataPar->outputs_count.emplace_back(gathered_vector.size());
  }

  chistov_a_gather_my::Sorting<int> testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  boost::mpi::gather(world, local_vector.data(), count_size_vector, vector, 0);
  if (world.rank() == 0) {
    std::sort(vector.begin(), vector.end());
    ASSERT_EQ(gathered_vector, vector);
  }
}

TEST(chistov_a_gather_my, test_count_is_a_prime_number_sort) {
  boost::mpi::communicator world;
  const int count_size_vector = 3;
  std::vector<int> local_vector;
  std::vector<int> vector;
  std::vector<int> gathered_vector(count_size_vector * world.size());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  local_vector = chistov_a_gather_my::getRandomVector<int>(count_size_vector);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(local_vector.data()));
  taskDataPar->inputs_count.emplace_back(count_size_vector);

  if (world.rank() == 0) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_vector.data()));
    taskDataPar->outputs_count.emplace_back(gathered_vector.size());
  }

  chistov_a_gather_my::Sorting<int> testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  boost::mpi::gather(world, local_vector.data(), count_size_vector, vector, 0);
  if (world.rank() == 0) {
    std::sort(vector.begin(), vector.end());
    ASSERT_EQ(gathered_vector, vector);
  }
}