#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/sedova_o_vertical_ribbon_scheme/include/ops_mpi.hpp"

//TEST(sedova_o_vertical_ribbon_scheme_mpi, Test_0) {
//  boost::mpi::communicator world;
//  std::vector<int> global_matrix;
//  std::vector<int> global_vector;
//  std::vector<int> global_result;
//
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//
//  if (world.rank() == 0) {
//    global_vector = {0, 0, 0};
//
//    global_result.resize(0, 0);
//
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
//    taskDataPar->inputs_count.emplace_back(global_matrix.size());
//
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
//    taskDataPar->inputs_count.emplace_back(global_vector.size());
//
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
//    taskDataPar->outputs_count.emplace_back(global_result.size());
//  }
//
//  auto taskParallel = std::make_shared<sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI>(taskDataPar);
//  EXPECT_FALSE(taskParallel->validation());
//}
//
//TEST(sedova_o_vertical_ribbon_scheme_mpi, Test_1) {
//  boost::mpi::communicator world;
//
//  int rows1 = 4;
//  int cols = 15;
//  int rows = 10;
//  std::vector<int> global_matrix;
//  std::vector<int> global_vector;
//  std::vector<int> global_result;
//
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//
//  if (world.rank() == 0) {
//    global_matrix.resize(rows * cols);
//    global_vector.resize(rows1);
//    global_result.resize(cols, 0);
//
//    for (int i = 0; i < rows * cols; ++i) {
//      global_matrix[i] = (rand() % 1000) - 500;
//    }
//    for (int i = 0; i < rows1; ++i) {
//      global_vector[i] = (rand() % 1000) - 500;
//    }
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
//    taskDataPar->inputs_count.emplace_back(global_matrix.size());
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
//    taskDataPar->inputs_count.emplace_back(global_vector.size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
//    taskDataPar->outputs_count.emplace_back(global_result.size());
//  }
//
//  auto taskParallel = std::make_shared<sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI>(taskDataPar);
//  ASSERT_FALSE(taskParallel->validation());
//}

TEST(sedova_o_vertical_ribbon_scheme_mpi, Test_2) {
  boost::mpi::communicator world;

  int rows1 = 15;
  int cols = 15;
  int rows = 10;
  std::vector<int> global_matrix;
  std::vector<int> global_vector;
  std::vector<int> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix.resize(rows * cols);
    global_vector.resize(rows1);
    global_result.resize(cols, 0);

    for (int i = 0; i < rows * cols; ++i) {
      global_matrix[i] = (rand() % 1000) - 500;
    }
    for (int i = 0; i < rows1; ++i) {
      global_vector[i] = (rand() % 1000) - 500;
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
}

TEST(sedova_o_vertical_ribbon_scheme_mpi, Test_3) {
  int rows_ = 5;
  int cols_ = 3;
  int count_proc = 5;
  std::vector<int> proc_(count_proc, 0);
  std::vector<int> off(count_proc, 0);
  if (count_proc > rows_) {
    for (int i = 0; i < rows_; ++i) {
      off[i] = i * cols_;
      proc_[i] = cols_;
    }
    for (int i = rows_; i < count_proc; ++i) {
      off[i] = -1;
      proc_[i] = 0;
    }
  } else {
    int count_proc_ = rows_ / count_proc;
    int surplus = rows_ % count_proc;
    int offset = 0;
    for (int i = 0; i < count_proc; ++i) {
      if (surplus > 0) {
        proc_[i] = (count_proc_ + 1) * cols_;
        --surplus;
      } else {
        proc_[i] = count_proc_ * cols_;
      }
      off[i] = offset;
      offset += proc_[i];
    }
  }
  std::vector<int> expected_proc = {3, 3, 3, 3, 3};
  std::vector<int> expected_off = {0, 3, 6, 9, 12};
  EXPECT_EQ(proc_, expected_proc);
  EXPECT_EQ(off, expected_off);
}

TEST(sedova_o_vertical_ribbon_scheme_mpi, Test_4) {
  int rows_ = 5;
  int cols_ = 3;
  int count_proc = 3;
  std::vector<int> proc_(count_proc, 0);
  std::vector<int> off(count_proc, 0);
  if (count_proc > rows_) {
    for (int i = 0; i < rows_; ++i) {
      off[i] = i * cols_;
      proc_[i] = cols_;
    }
    for (int i = rows_; i < count_proc; ++i) {
      off[i] = -1;
      proc_[i] = 0;
    }
  } else {
    int count_proc_ = rows_ / count_proc;
    int surplus = rows_ % count_proc;
    int offset = 0;
    for (int i = 0; i < count_proc; ++i) {
      if (surplus > 0) {
        proc_[i] = (count_proc_ + 1) * cols_;
        --surplus;
      } else {
        proc_[i] = count_proc_ * cols_;
      }
      off[i] = offset;
      offset += proc_[i];
    }
  }
  std::vector<int> expected_proc = {6, 6, 3};
  std::vector<int> expected_off = {0, 6, 12};
  EXPECT_EQ(proc_, expected_proc);
  EXPECT_EQ(off, expected_off);
}

TEST(sedova_o_vertical_ribbon_scheme_mpi, Test_5) {
  boost::mpi::communicator world;
  int rows_ = 10;
  int cols_ = 10;
  std::vector<int> global_matrix;
  std::vector<int> global_vector;
  std::vector<int> global_result;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_matrix.resize(rows_ * cols_);
    global_vector.resize(rows_);
    global_result.resize(cols_, 0);
    for (int i = 0; i < rows_ * cols_; ++i) {
      global_matrix[i] = (rand() % 1000) - 500;
    }
    for (int i = 0; i < rows_; ++i) {
      global_vector[i] = (rand() % 1000) - 500;
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }
  auto taskParallel = std::make_shared<sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  ASSERT_TRUE(taskParallel->pre_processing());
  ASSERT_TRUE(taskParallel->run());
  ASSERT_TRUE(taskParallel->post_processing());
}