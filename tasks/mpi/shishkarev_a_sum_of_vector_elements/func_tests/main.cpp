// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <numeric>  // Для std::accumulate
#include <vector>

#include "mpi/shishkarev_a_sum_of_vector_elements/include/ops_mpi.hpp"

TEST(shishkarev_a_sum_of_vector_elements_mpi, test_empty_sum) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  // Пустой вектор
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Для пустого вектора inputs и inputs_count корректно инициализируются
    taskDataPar->inputs.emplace_back(nullptr);  // Явно передаем nullptr
    taskDataPar->inputs_count.emplace_back(0);  // Указываем, что данных нет
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel parallel(taskDataPar);
  ASSERT_TRUE(parallel.validation());
  parallel.pre_processing();
  parallel.run();
  parallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_sum[0], 0);  // Проверяем, что результат равен 0
  }
}

TEST(shishkarev_a_sum_of_vector_elements_mpi, test_single_element_sum) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> global_vec(1, 42);
  std::vector<int32_t> global_sum(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel parallel(taskDataPar);
  ASSERT_TRUE(parallel.validation());
  parallel.pre_processing();
  parallel.run();
  parallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_sum[0], 42);
  }
}

TEST(shishkarev_a_sum_of_vector_elements_mpi, test_large_vector_sum) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const int vector_size = 1000000;
  if (world.rank() == 0) {
    global_vec = shishkarev_a_sum_of_vector_elements_mpi::getRandomVector(vector_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel parallel(taskDataPar);
  ASSERT_TRUE(parallel.validation());
  parallel.pre_processing();
  parallel.run();
  parallel.post_processing();

  if (world.rank() == 0) {
    int expected_sum = std::accumulate(global_vec.begin(), global_vec.end(), 0);
    ASSERT_EQ(global_sum[0], expected_sum);
  }
}

TEST(shishkarev_a_sum_of_vector_elements_mpi, test_zero_vector_sum) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> global_vec(100, 0);
  std::vector<int32_t> global_sum(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel parallel(taskDataPar);
  ASSERT_TRUE(parallel.validation());
  parallel.pre_processing();
  parallel.run();
  parallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_sum[0], 0);
  }
}

TEST(shishkarev_a_sum_of_vector_elements_mpi, test_negative_vector_sum) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> global_vec(100, -1);  // Вектор с отрицательными числами
  std::vector<int32_t> global_sum(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel parallel(taskDataPar);
  ASSERT_TRUE(parallel.validation());
  parallel.pre_processing();
  parallel.run();
  parallel.post_processing();

  if (world.rank() == 0) {
    int expected_sum = std::accumulate(global_vec.begin(), global_vec.end(), 0);  // Сумма: -100
    ASSERT_EQ(global_sum[0], expected_sum);
  }
}
