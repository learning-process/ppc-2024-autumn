#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/sedova_o_vertical_ribbon_scheme/include/ops_mpi.hpp"

void get_random_matrix(std::vector<int> &matrix, int a, int b) {
  std::random_device dev;
  std::mt19937 gen(dev());

  if (a >= b) {
    throw std::invalid_argument("error");
  }
  std::uniform_int_distribution<> dis(a, b);
  for (size_t i = 0; i < matrix.size(); ++i) {
    matrix[i] = dis(gen);
  }
}

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
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI TestSequential(taskDataSeq);
  EXPECT_FALSE(TestSequential.validation());
}

TEST(sedova_o_vertical_ribbon_scheme_mpi, correct_matrix_and_vector) {
  std::vector<int> matrix = {1, 2, 3, 4, 5, 6};
  std::vector<int> vector = {7, 8};
  std::vector<int> result(3, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI TestSequential(taskDataSeq);
  ASSERT_EQ(TestSequential.validation(), true);
  TestSequential.pre_processing();
  TestSequential.run();
  TestSequential.post_processing();

  std::vector<int> expected_result = {39, 54, 69};
  ASSERT_EQ(result, expected_result);
}

TEST(sedova_o_vertical_ribbon_scheme_mpi, mpi_and_seq) {
  boost::mpi::communicator world;

  std::vector<int> matrix;
  std::vector<int> vector;

  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(10);
    vector.resize(2);
    get_random_matrix(matrix, -10, 10);
    get_random_matrix(vector, -20, 2);

    result.resize(5, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    taskDataPar->inputs_count.emplace_back(vector.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_result(5, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    taskDataSeq->inputs_count.emplace_back(vector.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_result.data()));
    taskDataSeq->outputs_count.emplace_back(expected_result.size());

    // Create Task
    sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI testSequential(taskDataSeq);
    ASSERT_TRUE(testSequential.validation());
    testSequential.pre_processing();
    testSequential.run();
    testSequential.post_processing();

    ASSERT_EQ(result, expected_result);
  }
}