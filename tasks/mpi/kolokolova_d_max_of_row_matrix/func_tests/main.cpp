// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/kolokolova_d_max_of_row_matrix/include/ops_mpi.hpp"

TEST(kolokolova_d_max_of_row_matrix_mpi, Test_Parallel_Max1) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_max(world.size(), 0);
  int count_rows = world.size();
  std::cout << "Rang in test: " << world.rank() << "\n";
  std::cout << "Nums of procers in test: " << world.size() << "\n";
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = world.size() * 3;
    global_vec = kolokolova_d_max_of_row_matrix_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  kolokolova_d_max_of_row_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "+");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  //ASSERT_EQ(1, 1);

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(world.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
    taskDataSeq->inputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    kolokolova_d_max_of_row_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (int i = 0; i < int(reference_max.size()); i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
    
  }
}
  //TEST(kolokolova_d_max_of_vector_elements_mpi, Test_Parallel_Max2) {
//  boost::mpi::communicator world;
//  int num_processes = world.size();               // ���������� ���������� ���������
//  std::vector<int> global_max(num_processes, 0);  // ��������� ���������
//  int size_rows = num_processes * 4; // ������ ����
//  int count_rows = num_processes;
//  std::vector<int> global_mat(size_rows);  // �������
//  // Create TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//
//  if (world.rank() == 0) {
//    for (int i = 0; i < size_rows; ++i) {
//      global_mat[i] = i;
//    }
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
//    taskDataPar->inputs_count.emplace_back(global_mat.size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
//    taskDataPar->outputs_count.emplace_back(global_max.size());
//  }
//  kolokolova_d_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, num_processes);
//  ASSERT_EQ(testMpiTaskParallel.validation(), true);
//  testMpiTaskParallel.pre_processing();
//  testMpiTaskParallel.run();
//  testMpiTaskParallel.post_processing();
//
//  if (world.rank() == 0) {
//    std::vector<int> reference_max(num_processes, 0);
//    
//    // �������� TaskData
//    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
//    taskDataSeq->inputs_count.emplace_back(global_mat.size());
//    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
//    taskDataSeq->inputs_count.emplace_back((size_t)1);
//    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
//    taskDataSeq->outputs_count.emplace_back(reference_max.size());
//
//    // Create Task
//    kolokolova_d_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
//    ASSERT_EQ(testMpiTaskSequential.validation(), true);
//    testMpiTaskSequential.pre_processing();
//    testMpiTaskSequential.run();
//    testMpiTaskSequential.post_processing();
//
//    std::vector<int> results(num_processes);
//    std::memcpy(results.data(), taskDataPar->outputs.data(), num_processes * sizeof(int));
//
//    // ��������� �������� �����������
//    for (int i = 0; i < num_processes; ++i) {
//      EXPECT_EQ(results[i], reference_max[i]);
//    }
//  }
//}
//
//TEST(kolokolova_d_max_of_vector_elements_mpi, Test_Parallel_Max3) {
//  boost::mpi::communicator world;
//  int num_processes = world.size();               // ���������� ���������� ���������
//  int size_rows = num_processes * 10;
//  std::vector<int> global_max(num_processes, 0);  // ��������� ���������
//  std::vector<int> global_mat(size_rows);     // �������
//  int count_rows = num_processes;
//  // Create TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//
//  if (world.rank() == 0) {
//    for (int i = 0; i < size_rows; ++i) {
//      global_mat[i] = int(i+1);
//    }
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
//    taskDataPar->inputs_count.emplace_back(global_mat.size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
//    taskDataPar->outputs_count.emplace_back(global_max.size());
//  }
//  kolokolova_d_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, num_processes);
//  ASSERT_EQ(testMpiTaskParallel.validation(), true);
//  testMpiTaskParallel.pre_processing();
//  testMpiTaskParallel.run();
//  testMpiTaskParallel.post_processing();
//
//  if (world.rank() == 0) {
//    std::vector<int> reference_max(num_processes, 1);
//
//    // �������� TaskData
//    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
//    taskDataSeq->inputs_count.emplace_back(global_mat.size());
//    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
//    taskDataSeq->inputs_count.emplace_back((size_t)1);
//    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
//    taskDataSeq->outputs_count.emplace_back(reference_max.size());
//
//    // Create Task
//    kolokolova_d_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
//    ASSERT_EQ(testMpiTaskSequential.validation(), true);
//    testMpiTaskSequential.pre_processing();
//    testMpiTaskSequential.run();
//    testMpiTaskSequential.post_processing();
//
//    std::vector<int> results(num_processes);
//    std::memcpy(results.data(), taskDataPar->outputs.data(), num_processes * sizeof(int));
//
//    // ��������� �������� �����������
//    for (int i = 0; i < num_processes; ++i) {
//      EXPECT_EQ(results[i], reference_max[i]);
//    }
//  }
//}
//
//TEST(kolokolova_d_max_of_vector_elements_mpi, Test_Parallel_Max4) {
//  boost::mpi::communicator world;
//  int num_processes = world.size();               // ���������� ���������� ���������
//  int size_rows = num_processes * 10;
//  std::vector<int> global_max(num_processes, 0);  // ��������� ���������
//  std::vector<int> global_mat(size_rows);     // �������
//  int count_rows = num_processes;
//  // Create TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//
//  if (world.rank() == 0) {
//    for (int i = 0; i < size_rows; ++i) {
//      global_mat[i] = i+i;
//    }
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
//    taskDataPar->inputs_count.emplace_back(global_mat.size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
//    taskDataPar->outputs_count.emplace_back(global_max.size());
//  }
//  kolokolova_d_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, num_processes);
//  ASSERT_EQ(testMpiTaskParallel.validation(), true);
//  testMpiTaskParallel.pre_processing();
//  testMpiTaskParallel.run();
//  testMpiTaskParallel.post_processing();
//
//  if (world.rank() == 0) {
//    std::vector<int> reference_max(num_processes, 1);
//
//    // �������� TaskData
//    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
//    taskDataSeq->inputs_count.emplace_back(global_mat.size());
//    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
//    taskDataSeq->inputs_count.emplace_back((size_t)1);
//    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
//    taskDataSeq->outputs_count.emplace_back(reference_max.size());
//
//    // Create Task
//    kolokolova_d_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
//    ASSERT_EQ(testMpiTaskSequential.validation(), true);
//    testMpiTaskSequential.pre_processing();
//    testMpiTaskSequential.run();
//    testMpiTaskSequential.post_processing();
//
//    std::vector<int> results(num_processes);
//    std::memcpy(results.data(), taskDataPar->outputs.data(), num_processes * sizeof(int));
//
//    // ��������� �������� �����������
//    for (int i = 0; i < num_processes; ++i) {
//      EXPECT_EQ(results[i], reference_max[i]);
//      //std::cout << results[i] << " " << reference_max[i] << "\n";
//    }
//  }
//}
//
//TEST(kolokolova_d_max_of_vector_elements_mpi, Test_Parallel_Max5) {
//  boost::mpi::communicator world;
//  int num_processes = world.size();               // ���������� ���������� ���������
//  int size_rows = num_processes * 10;
//  std::vector<int> global_max(num_processes, 0);  // ��������� ���������
//  std::vector<int> global_mat(size_rows);     // �������
//  int count_rows = num_processes;
//  // Create TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//
//  if (world.rank() == 0) {
//    for (int i = 0; i < size_rows; ++i) {
//      global_mat[i] = int(i + i + i);
//    }
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
//    taskDataPar->inputs_count.emplace_back(global_mat.size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
//    taskDataPar->outputs_count.emplace_back(global_max.size());
//  }
//  kolokolova_d_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, num_processes);
//  ASSERT_EQ(testMpiTaskParallel.validation(), true);
//  testMpiTaskParallel.pre_processing();
//  testMpiTaskParallel.run();
//  testMpiTaskParallel.post_processing();
//
//  if (world.rank() == 0) {
//    std::vector<int> reference_max(num_processes, 1);
//
//    // �������� TaskData
//    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
//    taskDataSeq->inputs_count.emplace_back(global_mat.size());
//    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
//    taskDataSeq->inputs_count.emplace_back((size_t)1);
//    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
//    taskDataSeq->outputs_count.emplace_back(reference_max.size());
//
//    // Create Task
//    kolokolova_d_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
//    ASSERT_EQ(testMpiTaskSequential.validation(), true);
//    testMpiTaskSequential.pre_processing();
//    testMpiTaskSequential.run();
//    testMpiTaskSequential.post_processing();
//
//    std::vector<int> results(num_processes);
//    std::memcpy(results.data(), taskDataPar->outputs.data(), num_processes * sizeof(int));
//
//    // ��������� �������� �����������
//    for (int i = 0; i < num_processes; ++i) {
//      EXPECT_EQ(results[i], reference_max[i]);
//      // std::cout << results[i] << " " << reference_max[i] << "\n";
//    }
//  }
//}