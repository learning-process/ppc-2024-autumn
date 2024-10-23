// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/Shurygin_S_max_po_stolbam_matrix/include/ops_mpi.hpp"

TEST(Shurygin_S_max_po_stolbam_matrix_mpi, EmptyInputs) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(Shurygin_S_max_po_stolbam_matrix_mpi, EmptyOutputs) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    taskDataPar->inputs_count.push_back(3);
    taskDataPar->inputs_count.push_back(4);
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(new int[12]));
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(Shurygin_S_max_po_stolbam_matrix_mpi, IncorrectInputsCountSize) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    taskDataPar->inputs_count.push_back(3);  // Только одно значение в inputs_count
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(new int[12]));
    taskDataPar->outputs_count.push_back(4);
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(Shurygin_S_max_po_stolbam_matrix_mpi, IncorrectInputsCountValue) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    taskDataPar->inputs_count.push_back(3);
    taskDataPar->inputs_count.push_back(0);  // Число столбцов равно 0
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(new int[12]));
    taskDataPar->outputs_count.push_back(4);
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(Shurygin_S_max_po_stolbam_matrix_mpi, IncorrectOutputsCountSize) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    taskDataPar->inputs_count.push_back(3);
    taskDataPar->inputs_count.push_back(4);
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(new int[12]));
    taskDataPar->outputs_count.push_back(3);  // Неверное число элементов в outputs_count
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(Shurygin_S_max_po_stolbam_matrix_mpi, IncorrectOutputsCountValue) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    taskDataPar->inputs_count.push_back(3);
    taskDataPar->inputs_count.push_back(4);
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(new int[12]));
    taskDataPar->outputs_count.push_back(5);  // Неверное число элементов в outputs_count
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(Shurygin_S_max_po_stolbam_matrix_mpi, find_max_val_in_col_10x10_matrix) {
  boost::mpi::communicator world;
  const int count_rows = 10;
  const int count_columns = 10;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(count_columns, INT_MIN);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_matrix =
        Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskSequential::generate_random_matrix(count_rows, count_columns);
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataPar->inputs_count = {count_rows, count_columns};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int> reference_max(count_columns, INT_MIN);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataSeq->inputs_count = {count_rows, count_columns};
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());
    Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (int j = 0; j < count_columns; j++) {
      ASSERT_EQ(global_max[j], 200);
    }
  }
}

TEST(Shurygin_S_max_po_stolbam_matrix_mpi, find_max_val_in_col_100x100_matrix) {
  boost::mpi::communicator world;
  const int count_rows = 100;
  const int count_columns = 100;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(count_columns, INT_MIN);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_matrix =
        Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskSequential::generate_random_matrix(count_rows, count_columns);
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataPar->inputs_count = {count_rows, count_columns};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int> reference_max(count_columns, INT_MIN);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataSeq->inputs_count = {count_rows, count_columns};
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());
    Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (int j = 0; j < count_columns; j++) {
      ASSERT_EQ(global_max[j], 200);
    }
  }
}

TEST(Shurygin_S_max_po_stolbam_matrix_mpi, find_max_val_in_col_100x500_matrix) {
  boost::mpi::communicator world;
  const int count_rows = 100;
  const int count_columns = 500;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(count_columns, INT_MIN);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_matrix =
        Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskSequential::generate_random_matrix(count_rows, count_columns);
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataPar->inputs_count = {count_rows, count_columns};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int> reference_max(count_columns, INT_MIN);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataSeq->inputs_count = {count_rows, count_columns};
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());
    Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (int j = 0; j < count_columns; j++) {
      ASSERT_EQ(global_max[j], 200);
    }
  }
}

TEST(Shurygin_S_max_po_stolbam_matrix_mpi, find_max_val_in_col_3000x3000_matrix) {
  boost::mpi::communicator world;
  const int count_rows = 3000;
  const int count_columns = 3000;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(count_columns, INT_MIN);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_matrix =
        Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskSequential::generate_random_matrix(count_rows, count_columns);
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataPar->inputs_count = {count_rows, count_columns};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int> reference_max(count_columns, INT_MIN);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataSeq->inputs_count = {count_rows, count_columns};
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());
    Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (int j = 0; j < count_columns; j++) {
      ASSERT_EQ(global_max[j], 200);
    }
  }
}