// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/naumov_b_min_colum_matrix/include/ops_mpi.hpp"

TEST(naumov_b_min_colum_matrix_mpi, Test_Min_Column) {
  boost::mpi::communicator world;
  const int rows = 12;
  const int cols = 12;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int> global_minima(cols, std::numeric_limits<int>::max());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix.resize(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
      global_matrix[i] = naumov_b_min_colum_matrix_mpi::getRandomVector(cols);
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_minima.data()));
    taskDataPar->outputs_count.emplace_back(global_minima.size());
  }

  naumov_b_min_colum_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_minima(cols, std::numeric_limits<int>::max());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_minima.data()));
    taskDataSeq->outputs_count.emplace_back(reference_minima.size());

    naumov_b_min_colum_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_minima, global_minima);
  }
}

// Другие тесты
TEST(naumov_b_min_colum_matrix_mpi, Test_Min_Column_10x10) {
  boost::mpi::communicator world;
  const int rows = 10;
  const int cols = 10;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int> global_minima(cols, std::numeric_limits<int>::max());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix.resize(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
      global_matrix[i] = naumov_b_min_colum_matrix_mpi::getRandomVector(cols);
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_minima.data()));
    taskDataPar->outputs_count.emplace_back(global_minima.size());
  }

  naumov_b_min_colum_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_minima(cols, std::numeric_limits<int>::max());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_minima.data()));
    taskDataSeq->outputs_count.emplace_back(reference_minima.size());

    naumov_b_min_colum_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_minima, global_minima);
  }
}

TEST(naumov_b_min_colum_matrix_mpi, Test_Min_Column_8x15) {
  boost::mpi::communicator world;
  const int rows = 8;
  const int cols = 15;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int> global_minima(cols, std::numeric_limits<int>::max());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix.resize(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
      global_matrix[i] = naumov_b_min_colum_matrix_mpi::getRandomVector(cols);
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_minima.data()));
    taskDataPar->outputs_count.emplace_back(global_minima.size());
  }

  naumov_b_min_colum_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_minima(cols, std::numeric_limits<int>::max());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_minima.data()));
    taskDataSeq->outputs_count.emplace_back(reference_minima.size());

    naumov_b_min_colum_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_minima, global_minima);
  }
}

TEST(naumov_b_min_colum_matrix_mpi, Test_Min_Column_5x20) {
  boost::mpi::communicator world;
  const int rows = 5;
  const int cols = 20;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int> global_minima(cols, std::numeric_limits<int>::max());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix.resize(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
      global_matrix[i] = naumov_b_min_colum_matrix_mpi::getRandomVector(cols);
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_minima.data()));
    taskDataPar->outputs_count.emplace_back(global_minima.size());
  }

  naumov_b_min_colum_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_minima(cols, std::numeric_limits<int>::max());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_minima.data()));
    taskDataSeq->outputs_count.emplace_back(reference_minima.size());

    naumov_b_min_colum_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_minima, global_minima);
  }
}

TEST(naumov_b_min_colum_matrix_mpi, Test_Invalid_Input_Negative_Rows) {
  boost::mpi::communicator world;
  const int rows = -5;  // Неверное количество строк
  const int cols = 10;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int> global_minima(cols, std::numeric_limits<int>::max());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Создание матрицы с отрицательным количеством строк вызовет ошибку, так что просто не инициализируем global_matrix
    // global_matrix.resize(rows, std::vector<int>(cols));  // Удалено, чтобы избежать ошибки при выделении памяти
    // Вместо этого заполняем taskDataPar без инициализации матрицы
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_minima.data()));
    taskDataPar->outputs_count.emplace_back(global_minima.size());
  }

  naumov_b_min_colum_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  // Ожидаем провал валидации из-за отрицательного количества строк
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
}

TEST(naumov_b_min_colum_matrix_mpi, Test_Invalid_Input_Zero_Cols) {
  boost::mpi::communicator world;
  const int rows = 10;
  const int cols = 0;  // Неверное количество столбцов
  std::vector<std::vector<int>> global_matrix;
  std::vector<int> global_minima(cols, std::numeric_limits<int>::max());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix.resize(rows, std::vector<int>(cols));  // Попытка создать матрицу с нулевым количеством столбцов
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_minima.data()));
    taskDataPar->outputs_count.emplace_back(global_minima.size());
  }

  naumov_b_min_colum_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);  // Ожидаем провал валидации
}

// TEST(naumov_b_min_colum_matrix_mpi, Test_Sum) {
//   boost::mpi::communicator world;
//   std::vector<int> global_vec;
//   std::vector<int32_t> global_sum(1, 0);
//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

//   if (world.rank() == 0) {
//     const int count_size_vector = 120;
//     global_vec = naumov_b_min_colum_matrix_mpi::getRandomVector(count_size_vector);
//     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//     taskDataPar->inputs_count.emplace_back(global_vec.size());
//     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
//     taskDataPar->outputs_count.emplace_back(global_sum.size());
//   }

//   naumov_b_min_colum_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "+");
//   ASSERT_EQ(testMpiTaskParallel.validation(), true);
//   testMpiTaskParallel.pre_processing();
//   testMpiTaskParallel.run();
//   testMpiTaskParallel.post_processing();

//   if (world.rank() == 0) {
//     // Create data
//     std::vector<int32_t> reference_sum(1, 0);

//     // Create TaskData
//     std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//     taskDataSeq->inputs_count.emplace_back(global_vec.size());
//     taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
//     taskDataSeq->outputs_count.emplace_back(reference_sum.size());

//     // Create Task
//     naumov_b_min_colum_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "+");
//     ASSERT_EQ(testMpiTaskSequential.validation(), true);
//     testMpiTaskSequential.pre_processing();
//     testMpiTaskSequential.run();
//     testMpiTaskSequential.post_processing();

//     ASSERT_EQ(reference_sum[0], global_sum[0]);
//   }
// }

// TEST(naumov_b_min_colum_matrix_mpi, Test_Diff) {
//   boost::mpi::communicator world;
//   std::vector<int> global_vec;
//   std::vector<int32_t> global_diff(1, 0);
//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

//   if (world.rank() == 0) {
//     const int count_size_vector = 240;
//     global_vec = naumov_b_min_colum_matrix_mpi::getRandomVector(count_size_vector);
//     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//     taskDataPar->inputs_count.emplace_back(global_vec.size());
//     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_diff.data()));
//     taskDataPar->outputs_count.emplace_back(global_diff.size());
//   }

//   naumov_b_min_colum_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "-");
//   ASSERT_EQ(testMpiTaskParallel.validation(), true);
//   testMpiTaskParallel.pre_processing();
//   testMpiTaskParallel.run();
//   testMpiTaskParallel.post_processing();

//   if (world.rank() == 0) {
//     // Create data
//     std::vector<int32_t> reference_diff(1, 0);

//     // Create TaskData
//     std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//     taskDataSeq->inputs_count.emplace_back(global_vec.size());
//     taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_diff.data()));
//     taskDataSeq->outputs_count.emplace_back(reference_diff.size());

//     // Create Task
//     naumov_b_min_colum_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "-");
//     ASSERT_EQ(testMpiTaskSequential.validation(), true);
//     testMpiTaskSequential.pre_processing();
//     testMpiTaskSequential.run();
//     testMpiTaskSequential.post_processing();

//     ASSERT_EQ(reference_diff[0], global_diff[0]);
//   }
// }

// TEST(naumov_b_min_colum_matrix_mpi, Test_Diff_2) {
//   boost::mpi::communicator world;
//   std::vector<int> global_vec;
//   std::vector<int32_t> global_diff(1, 0);
//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

//   if (world.rank() == 0) {
//     const int count_size_vector = 120;
//     global_vec = naumov_b_min_colum_matrix_mpi::getRandomVector(count_size_vector);
//     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//     taskDataPar->inputs_count.emplace_back(global_vec.size());
//     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_diff.data()));
//     taskDataPar->outputs_count.emplace_back(global_diff.size());
//   }

//   naumov_b_min_colum_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "-");
//   ASSERT_EQ(testMpiTaskParallel.validation(), true);
//   testMpiTaskParallel.pre_processing();
//   testMpiTaskParallel.run();
//   testMpiTaskParallel.post_processing();

//   if (world.rank() == 0) {
//     // Create data
//     std::vector<int32_t> reference_diff(1, 0);

//     // Create TaskData
//     std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//     taskDataSeq->inputs_count.emplace_back(global_vec.size());
//     taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_diff.data()));
//     taskDataSeq->outputs_count.emplace_back(reference_diff.size());

//     // Create Task
//     naumov_b_min_colum_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "-");
//     ASSERT_EQ(testMpiTaskSequential.validation(), true);
//     testMpiTaskSequential.pre_processing();
//     testMpiTaskSequential.run();
//     testMpiTaskSequential.post_processing();

//     ASSERT_EQ(reference_diff[0], global_diff[0]);
//   }
// }

// TEST(naumov_b_min_colum_matrix_mpi, Test_Max) {
//   boost::mpi::communicator world;
//   std::vector<int> global_vec;
//   std::vector<int32_t> global_max(1, 0);
//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

//   if (world.rank() == 0) {
//     const int count_size_vector = 240;
//     global_vec = naumov_b_min_colum_matrix_mpi::getRandomVector(count_size_vector);
//     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//     taskDataPar->inputs_count.emplace_back(global_vec.size());
//     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
//     taskDataPar->outputs_count.emplace_back(global_max.size());
//   }

//   naumov_b_min_colum_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "max");
//   ASSERT_EQ(testMpiTaskParallel.validation(), true);
//   testMpiTaskParallel.pre_processing();
//   testMpiTaskParallel.run();
//   testMpiTaskParallel.post_processing();

//   if (world.rank() == 0) {
//     // Create data
//     std::vector<int32_t> reference_max(1, 0);

//     // Create TaskData
//     std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//     taskDataSeq->inputs_count.emplace_back(global_vec.size());
//     taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
//     taskDataSeq->outputs_count.emplace_back(reference_max.size());

//     // Create Task
//     naumov_b_min_colum_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "max");
//     ASSERT_EQ(testMpiTaskSequential.validation(), true);
//     testMpiTaskSequential.pre_processing();
//     testMpiTaskSequential.run();
//     testMpiTaskSequential.post_processing();

//     ASSERT_EQ(reference_max[0], global_max[0]);
//   }
// }

// TEST(naumov_b_min_colum_matrix_mpi, Test_Max_2) {
//   boost::mpi::communicator world;
//   std::vector<int> global_vec;
//   std::vector<int32_t> global_max(1, 0);
//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

//   if (world.rank() == 0) {
//     const int count_size_vector = 120;
//     global_vec = naumov_b_min_colum_matrix_mpi::getRandomVector(count_size_vector);
//     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//     taskDataPar->inputs_count.emplace_back(global_vec.size());
//     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
//     taskDataPar->outputs_count.emplace_back(global_max.size());
//   }

//   naumov_b_min_colum_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "max");
//   ASSERT_EQ(testMpiTaskParallel.validation(), true);
//   testMpiTaskParallel.pre_processing();
//   testMpiTaskParallel.run();
//   testMpiTaskParallel.post_processing();

//   if (world.rank() == 0) {
//     // Create data
//     std::vector<int32_t> reference_max(1, 0);

//     // Create TaskData
//     std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//     taskDataSeq->inputs_count.emplace_back(global_vec.size());
//     taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
//     taskDataSeq->outputs_count.emplace_back(reference_max.size());

//     // Create Task
//     naumov_b_min_colum_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "max");
//     ASSERT_EQ(testMpiTaskSequential.validation(), true);
//     testMpiTaskSequential.pre_processing();
//     testMpiTaskSequential.run();
//     testMpiTaskSequential.post_processing();

//     ASSERT_EQ(reference_max[0], global_max[0]);
//   }
// }
