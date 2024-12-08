#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>

#include "mpi/dudchenko_o_sleeping_barber/include/ops_mpi.hpp"

// Тест валидации с некорректным количеством входных данных
TEST(dudchenko_o_sleeping_barber_mpi, Test_Validation1) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    taskDataPar->inputs_count = {0};  // Некорректное количество входных данных
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}

// Тест валидации с недостаточным количеством процессов
TEST(dudchenko_o_sleeping_barber_mpi, Test_Validation2) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    if (world.size() < 3) {
      taskDataPar->inputs_count = {1};  // Количество мест 1
      EXPECT_FALSE(testMpiTaskParallel.validation());  // Ожидаем ошибку
    }
  }
}

// Тест валидации с корректными данными
TEST(dudchenko_o_sleeping_barber_mpi, Test_Validation3) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    if (world.size() < 3) {
      taskDataPar->inputs_count = {1};  // Количество мест 1
      EXPECT_FALSE(testMpiTaskParallel.validation());  // Ожидаем ошибку
    } else {
      taskDataPar->inputs_count = {1};  // Количество мест 1
      EXPECT_TRUE(testMpiTaskParallel.validation());  // Ожидаем успех
    }
  }
}

// Тест полного цикла с 1 местом для ожидания
TEST(dudchenko_o_sleeping_barber_mpi, Test_End_To_End1) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int max_waiting_chairs = 1;
  int global_res = -1;

  taskDataPar->inputs_count.emplace_back(max_waiting_chairs);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_res));
  taskDataPar->outputs_count.emplace_back(sizeof(global_res));

  dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber testMpiTaskParallel(taskDataPar);

  if (world.size() < 3) {
    ASSERT_FALSE(testMpiTaskParallel.validation());  // Ожидаем ошибку из-за недостатка процессов
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());  // Ожидаем успех
    ASSERT_TRUE(testMpiTaskParallel.pre_processing());  // Ожидаем успешную подготовку
    ASSERT_TRUE(testMpiTaskParallel.run());  // Ожидаем успешный запуск
    ASSERT_TRUE(testMpiTaskParallel.post_processing());  // Ожидаем успешную обработку

    world.barrier();  // Синхронизация всех процессов

    if (world.rank() == 0) {
      ASSERT_EQ(global_res, 0);  // Ожидаем, что результат будет 0
    }
  }
}

// Тест полного цикла с 3 местами для ожидания
TEST(dudchenko_o_sleeping_barber_mpi, Test_End_To_End2) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int max_waiting_chairs = 3;
  int global_res = -1;

  taskDataPar->inputs_count.emplace_back(max_waiting_chairs);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_res));
  taskDataPar->outputs_count.emplace_back(sizeof(global_res));

  dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber testMpiTaskParallel(taskDataPar);

  if (world.size() < 3) {
    ASSERT_FALSE(testMpiTaskParallel.validation());  // Ожидаем ошибку из-за недостатка процессов
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());  // Ожидаем успех
    ASSERT_TRUE(testMpiTaskParallel.pre_processing());  // Ожидаем успешную подготовку
    ASSERT_TRUE(testMpiTaskParallel.run());  // Ожидаем успешный запуск
    ASSERT_TRUE(testMpiTaskParallel.post_processing());  // Ожидаем успешную обработку

    world.barrier();  // Синхронизация всех процессов

    if (world.rank() == 0) {
      ASSERT_EQ(global_res, 0);  // Ожидаем, что результат будет 0
    }
  }
}

// Тест полного цикла с 996 местами для ожидания
TEST(dudchenko_o_sleeping_barber_mpi, Test_End_To_End3) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int max_waiting_chairs = 996;
  int global_res = -1;

  taskDataPar->inputs_count.emplace_back(max_waiting_chairs);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_res));
  taskDataPar->outputs_count.emplace_back(sizeof(global_res));

  dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber testMpiTaskParallel(taskDataPar);

  if (world.size() < 3) {
    ASSERT_FALSE(testMpiTaskParallel.validation());  // Ожидаем ошибку из-за недостатка процессов
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());  // Ожидаем успех
    ASSERT_TRUE(testMpiTaskParallel.pre_processing());  // Ожидаем успешную подготовку
    ASSERT_TRUE(testMpiTaskParallel.run());  // Ожидаем успешный запуск
    ASSERT_TRUE(testMpiTaskParallel.post_processing());  // Ожидаем успешную обработку

    world.barrier();  // Синхронизация всех процессов

    if (world.rank() == 0) {
      ASSERT_EQ(global_res, 0);  // Ожидаем, что результат будет 0
    }
  }
}

// Тест полного цикла с 999 местами для ожидания
TEST(dudchenko_o_sleeping_barber_mpi, Test_End_To_End4) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int max_waiting_chairs = 999;
  int global_res = -1;

  taskDataPar->inputs_count.emplace_back(max_waiting_chairs);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_res));
  taskDataPar->outputs_count.emplace_back(sizeof(global_res));

  dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber testMpiTaskParallel(taskDataPar);

  if (world.size() < 3) {
    ASSERT_FALSE(testMpiTaskParallel.validation());  // Ожидаем ошибку из-за недостатка процессов
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());  // Ожидаем успех
    ASSERT_TRUE(testMpiTaskParallel.pre_processing());  // Ожидаем успешную подготовку
    ASSERT_TRUE(testMpiTaskParallel.run());  // Ожидаем успешный запуск
    ASSERT_TRUE(testMpiTaskParallel.post_processing());  // Ожидаем успешную обработку

    world.barrier();  // Синхронизация всех процессов

    if (world.rank() == 0) {
      ASSERT_EQ(global_res, 0);  // Ожидаем, что результат будет 0
    }
  }
}

// Тест полного цикла с 1024 местами для ожидания
TEST(dudchenko_o_sleeping_barber_mpi, Test_End_To_End5) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int max_waiting_chairs = 1024;
  int global_res = -1;

  taskDataPar->inputs_count.emplace_back(max_waiting_chairs);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_res));
  taskDataPar->outputs_count.emplace_back(sizeof(global_res));

  dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber testMpiTaskParallel(taskDataPar);

  if (world.size() < 3) {
    ASSERT_FALSE(testMpiTaskParallel.validation());  // Ожидаем ошибку из-за недостатка процессов
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());  // Ожидаем успех
    ASSERT_TRUE(testMpiTaskParallel.pre_processing());  // Ожидаем успешную подготовку
    ASSERT_TRUE(testMpiTaskParallel.run());  // Ожидаем успешный запуск
    ASSERT_TRUE(testMpiTaskParallel.post_processing());  // Ожидаем успешную обработку

    world.barrier();  // Синхронизация всех процессов

    if (world.rank() == 0) {
      ASSERT_EQ(global_res, 0);  // Ожидаем, что результат будет 0
    }
  }
}
