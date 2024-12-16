#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/sedova_o_vertical_ribbon_scheme/include/ops_mpi.hpp"

TEST(sedova_o_vertical_ribbon_scheme_mpi, distribution1) {
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

TEST(sedova_o_vertical_ribbon_scheme_mpi, distribution2) {
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

TEST(sedova_o_vertical_ribbon_scheme_mpi, distribution3) {
  int rows_ = 5;
  int cols_ = 4;
  int count_proc = 6;
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
  std::vector<int> expected_proc = {4, 4, 4, 4, 4, 0};
  std::vector<int> expected_off = {0, 4, 8, 12, 16, -1};
  EXPECT_EQ(proc_, expected_proc);
  EXPECT_EQ(off, expected_off);
}

TEST(sedova_o_vertical_ribbon_scheme_mpi, distribution4) {
  int rows_ = 10;
  int cols_ = 4;
  int count_proc = 8;
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
  std::vector<int> expected_proc = {8, 8, 4, 4, 4, 4, 4, 4};
  std::vector<int> expected_off = {0, 8, 16, 20, 24, 28, 32, 36};
  EXPECT_EQ(proc_, expected_proc);
  EXPECT_EQ(off, expected_off);
}

TEST(sedova_o_vertical_ribbon_scheme_mpi, false_validation) {
  std::vector<int> matrix = {1, 2, 3};
  std::vector<int> vector = {7, 8};
  std::vector<int> result(3, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI TestSequential(taskDataSeq);
  EXPECT_FALSE(TestSequential.validation());
}

TEST(sedova_o_vertical_ribbon_scheme_mpi, true_validation) {
  std::vector<int> matrix = {1, 2, 3, 4};
  std::vector<int> vector = {7, 8};
  std::vector<int> result(2, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataPar->inputs_count.emplace_back(matrix.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
  taskDataPar->inputs_count.emplace_back(vector.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataPar->outputs_count.emplace_back(result.size());

  sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI taskParallel(taskDataPar);
  EXPECT_TRUE(taskParallel.validation());
}

TEST(sedova_o_vertical_ribbon_scheme_mpi, correct_matrix_and_vector_seq) {
  std::vector<int> matrix = {1, 2, 3, 4, 5, 6};
  std::vector<int> vector = {7, 8};
  std::vector<int> result(3, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI TestSequential(taskDataSeq);
  ASSERT_TRUE(TestSequential.validation());
  TestSequential.pre_processing();
  TestSequential.run();
  TestSequential.post_processing();

  std::vector<int> expected_result = {39, 54, 69};
  ASSERT_EQ(result, expected_result);
}