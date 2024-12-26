// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/varfolomeev_g_matrix_max_rows_vals/include/ops_mpi.hpp"

namespace varfolomeev_g_matrix_max_rows_vals_mpi {

static void getRandomVectorBetween(std::vector<int> &vec, int a, int b) {  // [a, b]
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (size_t i = 0; i < vec.size(); i++) {
    vec[i] = std::rand() % (b - a + 1) + a;
  }
}
}  // namespace varfolomeev_g_matrix_max_rows_vals_mpi

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_Empty_Matrix) {
  int size_m = 0;
  int size_n = 0;

  boost::mpi::communicator world;

  std::vector<int> global_mat;
  std::vector<int32_t> global_max(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel maxInRowsParallel(taskDataPar);
  ASSERT_EQ(maxInRowsParallel.validation(), true);  // Валидация должна пройти успешно
  maxInRowsParallel.pre_processing();
  maxInRowsParallel.run();
  maxInRowsParallel.post_processing();
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_getRandomVectorBetween) {
  int sz = 100;
  std::vector<int> vec(sz);
  varfolomeev_g_matrix_max_rows_vals_mpi::getRandomVectorBetween(vec, -100, 100);

  // Проверка размера вектора
  ASSERT_EQ((int)vec.size(), sz);

  // Проверка, что все элементы находятся в диапазоне от -100 до 99
  for (int i = 0; i < sz; ++i) {
    ASSERT_GE(vec[i], -100);
    ASSERT_LE(vec[i], 100);
  }
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_gen_5x5_Matrix) {
  int size_m = 5;
  int size_n = 5;

  boost::mpi::communicator world;

  std::vector<int> global_mat(size_m * size_n);
  std::vector<int32_t> global_max(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    varfolomeev_g_matrix_max_rows_vals_mpi::getRandomVectorBetween(global_mat, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel maxInRowsParallel(taskDataPar);
  ASSERT_EQ(maxInRowsParallel.validation(), true);
  maxInRowsParallel.pre_processing();
  maxInRowsParallel.run();
  maxInRowsParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(size_m, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(size_m);
    taskDataSeq->inputs_count.emplace_back(size_n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsSequential maxInRowsSequential(taskDataSeq);
    ASSERT_EQ(maxInRowsSequential.validation(), true);
    maxInRowsSequential.pre_processing();
    maxInRowsSequential.run();
    maxInRowsSequential.post_processing();

    for (int i = 0; i < size_m; i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_gen_1x5_Matrix) {
  int size_m = 1;
  int size_n = 5;

  boost::mpi::communicator world;
  std::vector<int> global_mat(size_m * size_n);
  std::vector<int32_t> global_max(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    varfolomeev_g_matrix_max_rows_vals_mpi::getRandomVectorBetween(global_mat, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel maxInRowsParallel(taskDataPar);
  ASSERT_EQ(maxInRowsParallel.validation(), true);
  maxInRowsParallel.pre_processing();
  maxInRowsParallel.run();
  maxInRowsParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(size_m, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(size_m);
    taskDataSeq->inputs_count.emplace_back(size_n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsSequential maxInRowsSequential(taskDataSeq);
    ASSERT_EQ(maxInRowsSequential.validation(), true);
    maxInRowsSequential.pre_processing();
    maxInRowsSequential.run();
    maxInRowsSequential.post_processing();

    for (int i = 0; i < size_m; i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_gen_1x500_Matrix) {
  int size_m = 1;
  int size_n = 500;

  boost::mpi::communicator world;

  std::vector<int> global_mat(size_m * size_n);
  std::vector<int32_t> global_max(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    varfolomeev_g_matrix_max_rows_vals_mpi::getRandomVectorBetween(global_mat, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel maxInRowsParallel(taskDataPar);
  ASSERT_EQ(maxInRowsParallel.validation(), true);
  maxInRowsParallel.pre_processing();
  maxInRowsParallel.run();
  maxInRowsParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(size_m, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(size_m);
    taskDataSeq->inputs_count.emplace_back(size_n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsSequential maxInRowsSequential(taskDataSeq);
    ASSERT_EQ(maxInRowsSequential.validation(), true);
    maxInRowsSequential.pre_processing();
    maxInRowsSequential.run();
    maxInRowsSequential.post_processing();

    for (int i = 0; i < size_m; i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_gen_500x1_Matrix) {
  int size_m = 500;
  int size_n = 1;

  boost::mpi::communicator world;

  std::vector<int> global_mat(size_m * size_n);
  std::vector<int32_t> global_max(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    varfolomeev_g_matrix_max_rows_vals_mpi::getRandomVectorBetween(global_mat, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel maxInRowsParallel(taskDataPar);
  ASSERT_EQ(maxInRowsParallel.validation(), true);
  maxInRowsParallel.pre_processing();
  maxInRowsParallel.run();
  maxInRowsParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(size_m, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(size_m);
    taskDataSeq->inputs_count.emplace_back(size_n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsSequential maxInRowsSequential(taskDataSeq);
    ASSERT_EQ(maxInRowsSequential.validation(), true);
    maxInRowsSequential.pre_processing();
    maxInRowsSequential.run();
    maxInRowsSequential.post_processing();

    for (int i = 0; i < size_m; i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_gen_50x50_Matrix) {
  int size_m = 50;
  int size_n = 50;

  boost::mpi::communicator world;

  std::vector<int> global_mat(size_m * size_n);
  std::vector<int32_t> global_max(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    varfolomeev_g_matrix_max_rows_vals_mpi::getRandomVectorBetween(global_mat, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel maxInRowsParallel(taskDataPar);
  ASSERT_EQ(maxInRowsParallel.validation(), true);
  maxInRowsParallel.pre_processing();
  maxInRowsParallel.run();
  maxInRowsParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(size_m, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(size_m);
    taskDataSeq->inputs_count.emplace_back(size_n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsSequential maxInRowsSequential(taskDataSeq);
    ASSERT_EQ(maxInRowsSequential.validation(), true);
    maxInRowsSequential.pre_processing();
    maxInRowsSequential.run();
    maxInRowsSequential.post_processing();

    for (int i = 0; i < size_m; i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_gen_50x100_Matrix) {
  int size_m = 100;
  int size_n = 50;

  boost::mpi::communicator world;

  std::vector<int> global_mat(size_m * size_n);
  std::vector<int32_t> global_max(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    varfolomeev_g_matrix_max_rows_vals_mpi::getRandomVectorBetween(global_mat, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel maxInRowsParallel(taskDataPar);
  ASSERT_EQ(maxInRowsParallel.validation(), true);
  maxInRowsParallel.pre_processing();
  maxInRowsParallel.run();
  maxInRowsParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(size_m, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(size_m);
    taskDataSeq->inputs_count.emplace_back(size_n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsSequential maxInRowsSequential(taskDataSeq);
    ASSERT_EQ(maxInRowsSequential.validation(), true);
    maxInRowsSequential.pre_processing();
    maxInRowsSequential.run();
    maxInRowsSequential.post_processing();

    for (int i = 0; i < size_m; i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_gen_100x200_Matrix) {
  int size_m = 200;
  int size_n = 100;

  boost::mpi::communicator world;

  std::vector<int> global_mat(size_m * size_n);
  std::vector<int32_t> global_max(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    varfolomeev_g_matrix_max_rows_vals_mpi::getRandomVectorBetween(global_mat, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel maxInRowsParallel(taskDataPar);
  ASSERT_EQ(maxInRowsParallel.validation(), true);
  maxInRowsParallel.pre_processing();
  maxInRowsParallel.run();
  maxInRowsParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(size_m, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(size_m);
    taskDataSeq->inputs_count.emplace_back(size_n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsSequential maxInRowsSequential(taskDataSeq);
    ASSERT_EQ(maxInRowsSequential.validation(), true);
    maxInRowsSequential.pre_processing();
    maxInRowsSequential.run();
    maxInRowsSequential.post_processing();

    for (int i = 0; i < size_m; i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_gen_5000x5000_Matrix) {
  int size_m = 5000;
  int size_n = 5000;

  boost::mpi::communicator world;

  std::vector<int> global_mat(size_m * size_n);
  std::vector<int32_t> global_max(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    varfolomeev_g_matrix_max_rows_vals_mpi::getRandomVectorBetween(global_mat, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel maxInRowsParallel(taskDataPar);
  ASSERT_EQ(maxInRowsParallel.validation(), true);
  maxInRowsParallel.pre_processing();
  maxInRowsParallel.run();
  maxInRowsParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(size_m, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(size_m);
    taskDataSeq->inputs_count.emplace_back(size_n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsSequential maxInRowsSequential(taskDataSeq);
    ASSERT_EQ(maxInRowsSequential.validation(), true);
    maxInRowsSequential.pre_processing();
    maxInRowsSequential.run();
    maxInRowsSequential.post_processing();

    for (int i = 0; i < size_m; i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_manual_3x3_negative_Matrix) {
  int size_m = 3;
  int size_n = 3;

  boost::mpi::communicator world;

  std::vector<int> global_mat = {-1, -2, -3, -4, -5, -6, -7, -8, -9};
  std::vector<int32_t> global_max(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel maxInRowsParallel(taskDataPar);
  ASSERT_EQ(maxInRowsParallel.validation(), true);
  maxInRowsParallel.pre_processing();
  maxInRowsParallel.run();
  maxInRowsParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(size_m, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(size_m);
    taskDataSeq->inputs_count.emplace_back(size_n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsSequential maxInRowsSequential(taskDataSeq);
    ASSERT_EQ(maxInRowsSequential.validation(), true);
    maxInRowsSequential.pre_processing();
    maxInRowsSequential.run();
    maxInRowsSequential.post_processing();

    for (int i = 0; i < size_m; i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_500x500_negative_Matrix) {
  int size_m = 500;
  int size_n = 500;

  boost::mpi::communicator world;

  std::vector<int> global_mat(size_m * size_n);
  std::vector<int32_t> global_max(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    varfolomeev_g_matrix_max_rows_vals_mpi::getRandomVectorBetween(global_mat, -200, -1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel maxInRowsParallel(taskDataPar);
  ASSERT_EQ(maxInRowsParallel.validation(), true);
  maxInRowsParallel.pre_processing();
  maxInRowsParallel.run();
  maxInRowsParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(size_m, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(size_m);
    taskDataSeq->inputs_count.emplace_back(size_n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsSequential maxInRowsSequential(taskDataSeq);
    ASSERT_EQ(maxInRowsSequential.validation(), true);
    maxInRowsSequential.pre_processing();
    maxInRowsSequential.run();
    maxInRowsSequential.post_processing();

    for (int i = 0; i < size_m; i++) {
      ASSERT_LE(reference_max[i], 0);
    }
  }
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_300x300_zero_Matrix) {
  int size_m = 300;
  int size_n = 300;

  boost::mpi::communicator world;

  std::vector<int> global_mat(size_m * size_n);
  std::vector<int32_t> global_max(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    varfolomeev_g_matrix_max_rows_vals_mpi::getRandomVectorBetween(global_mat, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel maxInRowsParallel(taskDataPar);
  ASSERT_EQ(maxInRowsParallel.validation(), true);
  maxInRowsParallel.pre_processing();
  maxInRowsParallel.run();
  maxInRowsParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(size_m, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(size_m);
    taskDataSeq->inputs_count.emplace_back(size_n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsSequential maxInRowsSequential(taskDataSeq);
    ASSERT_EQ(maxInRowsSequential.validation(), true);
    maxInRowsSequential.pre_processing();
    maxInRowsSequential.run();
    maxInRowsSequential.post_processing();

    for (int i = 0; i < size_m; i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_manual_1x1_single_Matrix) {
  int size_m = 1;
  int size_n = 1;

  boost::mpi::communicator world;

  std::vector<int> global_mat = {42};
  std::vector<int32_t> global_max(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel maxInRowsParallel(taskDataPar);
  ASSERT_EQ(maxInRowsParallel.validation(), true);
  maxInRowsParallel.pre_processing();
  maxInRowsParallel.run();
  maxInRowsParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(size_m, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(size_m);
    taskDataSeq->inputs_count.emplace_back(size_n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsSequential maxInRowsSequential(taskDataSeq);
    ASSERT_EQ(maxInRowsSequential.validation(), true);
    maxInRowsSequential.pre_processing();
    maxInRowsSequential.run();
    maxInRowsSequential.post_processing();

    for (int i = 0; i < size_m; i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_manual_5x3_maxes_in_the_end) {
  int size_m = 5;
  int size_n = 3;

  boost::mpi::communicator world;

  std::vector<int> global_mat = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<int32_t> global_max(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel maxInRowsParallel(taskDataPar);
  ASSERT_EQ(maxInRowsParallel.validation(), true);
  maxInRowsParallel.pre_processing();
  maxInRowsParallel.run();
  maxInRowsParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(size_m, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(size_m);
    taskDataSeq->inputs_count.emplace_back(size_n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsSequential maxInRowsSequential(taskDataSeq);
    ASSERT_EQ(maxInRowsSequential.validation(), true);
    maxInRowsSequential.pre_processing();
    maxInRowsSequential.run();
    maxInRowsSequential.post_processing();

    for (int i = 0; i < size_m; i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_random_500x300_maxes_in_the_end) {
  int size_m = 500;
  int size_n = 300;
  boost::mpi::communicator world;

  std::vector<int32_t> global_mat(size_m * size_n);
  varfolomeev_g_matrix_max_rows_vals_mpi::getRandomVectorBetween(global_mat, -100, 100);
  for (int i = 0; i < size_m; ++i) {
    global_mat[i * size_n + (size_n - 1)] = 200;
  }
  std::vector<int32_t> global_max(size_m, 200);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel maxInRowsParallel(taskDataPar);
  ASSERT_EQ(maxInRowsParallel.validation(), true);
  maxInRowsParallel.pre_processing();
  maxInRowsParallel.run();
  maxInRowsParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(size_m, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(size_m);
    taskDataSeq->inputs_count.emplace_back(size_n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsSequential maxInRowsSequential(taskDataSeq);
    ASSERT_EQ(maxInRowsSequential.validation(), true);
    maxInRowsSequential.pre_processing();
    maxInRowsSequential.run();
    maxInRowsSequential.post_processing();

    for (int i = 0; i < size_m; i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}