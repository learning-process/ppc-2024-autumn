// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/varfolomeev_g_matrix_max_rows_vals/include/ops_mpi.hpp"

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_gen_5x5_Matrix) {
  int size_m = 5;
  int size_n = 5;

  boost::mpi::communicator world;

  std::vector<int> global_mat;
  std::vector<int32_t> global_max(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    global_mat = varfolomeev_g_matrix_max_rows_vals_mpi::getRandomVector(size_n * size_m);
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

    for (size_t i = 0; i < size_m; i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_gen_1x5_Matrix) {
  int size_m = 1;
  int size_n = 5;

  boost::mpi::communicator world;

  std::vector<int> global_mat;
  std::vector<int32_t> global_max(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    global_mat = varfolomeev_g_matrix_max_rows_vals_mpi::getRandomVector(size_n * size_m);
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

    for (size_t i = 0; i < size_m; i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_gen_1x5000_Matrix) {
  int size_m = 1;
  int size_n = 5000;

  boost::mpi::communicator world;

  std::vector<int> global_mat;
  std::vector<int32_t> global_max(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    global_mat = varfolomeev_g_matrix_max_rows_vals_mpi::getRandomVector(size_n * size_m);
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

    for (size_t i = 0; i < size_m; i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_gen_5000x1_Matrix) {
  int size_m = 5000;
  int size_n = 1;

  boost::mpi::communicator world;

  std::vector<int> global_mat;
  std::vector<int32_t> global_max(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    global_mat = varfolomeev_g_matrix_max_rows_vals_mpi::getRandomVector(size_n * size_m);
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

    for (size_t i = 0; i < size_m; i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_gen_50x50_Matrix) {
  int size_m = 50;
  int size_n = 50;

  boost::mpi::communicator world;

  std::vector<int> global_mat;
  std::vector<int32_t> global_max(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    global_mat = varfolomeev_g_matrix_max_rows_vals_mpi::getRandomVector(size_n * size_m);
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

    for (size_t i = 0; i < size_m; i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_gen_50x100_Matrix) {
  int size_m = 100;
  int size_n = 50;

  boost::mpi::communicator world;

  std::vector<int> global_mat;
  std::vector<int32_t> global_max(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    global_mat = varfolomeev_g_matrix_max_rows_vals_mpi::getRandomVector(size_n * size_m);
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

    for (size_t i = 0; i < size_m; i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_gen_100x200_Matrix) {
  int size_m = 200;
  int size_n = 100;

  boost::mpi::communicator world;

  std::vector<int> global_mat;
  std::vector<int32_t> global_max(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    global_mat = varfolomeev_g_matrix_max_rows_vals_mpi::getRandomVector(size_n * size_m);
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

    for (size_t i = 0; i < size_m; i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_gen_1000x1000_Matrix) {
  int size_m = 1000;
  int size_n = 1000;

  boost::mpi::communicator world;

  std::vector<int> global_mat;
  std::vector<int32_t> global_max(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    global_mat = varfolomeev_g_matrix_max_rows_vals_mpi::getRandomVector(size_n * size_m);
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

    for (size_t i = 0; i < size_m; i++) {
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

    for (size_t i = 0; i < size_m; i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_manual_3x3_zero_Matrix) {
  int size_m = 3;
  int size_n = 3;

  boost::mpi::communicator world;

  std::vector<int> global_mat = {0, 0, 0, 0, 0, 0, 0, 0, 0};
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

    for (size_t i = 0; i < size_m; i++) {
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

    for (size_t i = 0; i < size_m; i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(varfolomeev_g_matrix_max_rows_mpi, Test_manual_5x3_maxes_in_the_end_Matrix) {
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

    for (size_t i = 0; i < size_m; i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}