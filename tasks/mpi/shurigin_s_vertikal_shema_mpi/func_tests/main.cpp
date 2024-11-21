#include <gtest/gtest.h>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <memory>
#include <vector>

#include "mpi/shurigin_s_vertikal_shema_mpi/include/ops_mpi.hpp"

using namespace shurigin_s_vertikal_shema;

TEST(shurigin_s_vertikal_shema, calculate_distribution_num_proc_greater_than_cols) {
  int rows = 4;
  int cols = 3;
  int num_proc = 5;
  std::vector<int> sizes;
  std::vector<int> displs;

  calculate_distribution(rows, cols, num_proc, sizes, displs);

  ASSERT_EQ(static_cast<int>(sizes.size()), num_proc);
  ASSERT_EQ(static_cast<int>(displs.size()), num_proc);

  for (int i = 0; i < num_proc; ++i) {
    if (i < cols) {
      EXPECT_EQ(sizes[i], rows);
      EXPECT_EQ(displs[i], i * rows);
    } else {
      EXPECT_EQ(sizes[i], 0);
      EXPECT_EQ(displs[i], -1);
    }
  }
}

TEST(shurigin_s_vertikal_shema, calculate_distribution_num_proc_less_than_cols_with_remainder) {
  int rows = 3;
  int cols = 10;
  int num_proc = 4;
  std::vector<int> sizes;
  std::vector<int> displs;

  calculate_distribution(rows, cols, num_proc, sizes, displs);

  ASSERT_EQ(static_cast<int>(sizes.size()), num_proc);
  ASSERT_EQ(static_cast<int>(displs.size()), num_proc);

  int expected_sizes[] = {9, 9, 6, 6};
  int expected_displs[] = {0, 9, 18, 24};

  for (int i = 0; i < num_proc; ++i) {
    EXPECT_EQ(sizes[i], expected_sizes[i]);
    EXPECT_EQ(displs[i], expected_displs[i]);
  }
}

TEST(shurigin_s_vertikal_shema, calculate_distribution_num_proc_less_than_cols_no_remainder) {
  int rows = 2;
  int cols = 8;
  int num_proc = 4;
  std::vector<int> sizes;
  std::vector<int> displs;

  calculate_distribution(rows, cols, num_proc, sizes, displs);

  ASSERT_EQ(static_cast<int>(sizes.size()), num_proc);
  ASSERT_EQ(static_cast<int>(displs.size()), num_proc);

  int expected_sizes[] = {4, 4, 4, 4};
  int expected_displs[] = {0, 4, 8, 12};

  for (int i = 0; i < num_proc; ++i) {
    EXPECT_EQ(sizes[i], expected_sizes[i]);
    EXPECT_EQ(displs[i], expected_displs[i]);
  }
}

TEST(shurigin_s_vertikal_shema, empty_matrix_test) {
  boost::mpi::communicator world;

  std::vector<int> global_matrix;
  std::vector<int> global_vector;
  std::vector<int> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_rows = 0;

  if (world.rank() == 0) {
    global_vector = {1, 1, 1};

    global_result.resize(num_rows, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<TestTaskMPI>(taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel->validation());
  } else {
    EXPECT_TRUE(taskParallel->validation());
  }
}

TEST(shurigin_s_vertikal_shema, empty_vector_test) {
  boost::mpi::communicator world;

  std::vector<int> global_matrix;
  std::vector<int> global_vector;
  std::vector<int> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_rows = 2;

  if (world.rank() == 0) {
    global_matrix = {1, 1, 1, 1};

    global_result.resize(num_rows, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<TestTaskMPI>(taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel->validation());
  } else {
    EXPECT_TRUE(taskParallel->validation());
  }
}

TEST(shurigin_s_vertikal_shema, invalid_matrix_vector_size_test) {
  boost::mpi::communicator world;

  std::vector<int> global_matrix;
  std::vector<int> global_vector;
  std::vector<int> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_rows = 3;

  if (world.rank() == 0) {
    global_matrix = {1, 4, 7, 2, 5, 8, 3, 6, 9};
    global_vector = {1, 0};

    global_result.resize(num_rows, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<TestTaskMPI>(taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel->validation());
  } else {
    EXPECT_TRUE(taskParallel->validation());
  }
}

TEST(shurigin_s_vertikal_shema, invalid_result_size_test) {
  boost::mpi::communicator world;

  std::vector<int> global_matrix;
  std::vector<int> global_vector;
  std::vector<int> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {1, 3, 5, 2, 4, 6};
    global_vector = {1, 0};

    global_result.resize(5);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<TestTaskMPI>(taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel->validation());
  } else {
    EXPECT_TRUE(taskParallel->validation());
  }
}

TEST(shurigin_s_vertikal_shema, small_matrix_vector_parallel_test) {
  boost::mpi::communicator world;

  std::vector<int> global_matrix;
  std::vector<int> global_vector;
  std::vector<int> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_rows = 4;

  if (world.rank() == 0) {
    global_matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    global_vector = {1, 0, -1};

    global_result.resize(num_rows, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<TestTaskMPI>(taskDataPar);

  ASSERT_TRUE(taskParallel->validation());
  ASSERT_TRUE(taskParallel->pre_processing());
  ASSERT_TRUE(taskParallel->run());
  ASSERT_TRUE(taskParallel->post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_result = {-2, -2, -2, -2};
    ASSERT_EQ(global_result.size(), expected_result.size());
    for (size_t i = 0; i < global_result.size(); ++i) {
      ASSERT_EQ(global_result[i], expected_result[i]);
    }
  }
}

TEST(shurigin_s_vertikal_shema, small_matrix_vector_sequential_test) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    std::vector<int> global_matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int> global_vector = {1, 0, -1};
    std::vector<int> seq_result(4, 0);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataSeq->inputs_count.emplace_back(global_vector.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    taskDataSeq->outputs_count.emplace_back(seq_result.size());

    auto taskSequential = std::make_shared<TestTaskSequential>(taskDataSeq);

    ASSERT_TRUE(taskSequential->validation());
    ASSERT_TRUE(taskSequential->pre_processing());
    ASSERT_TRUE(taskSequential->run());
    ASSERT_TRUE(taskSequential->post_processing());

    std::vector<int> expected_result = {-2, -2, -2, -2};
    ASSERT_EQ(seq_result.size(), expected_result.size());
    for (size_t i = 0; i < seq_result.size(); ++i) {
      ASSERT_EQ(seq_result[i], expected_result[i]);
    }
  }
}

TEST(shurigin_s_vertikal_shema, prime_size_matrix_test) {
  boost::mpi::communicator world;

  std::vector<int> global_matrix;
  std::vector<int> global_vector;
  std::vector<int> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_rows = 7;   
  int num_cols = 11;   

  if (world.rank() == 0) {
    global_matrix.resize(num_rows * num_cols);
    for (int i = 0; i < num_rows * num_cols; ++i) {
      global_matrix[i] = i + 1;
    }

    global_vector.resize(num_cols);
    for (int i = 0; i < num_cols; ++i) {
      global_vector[i] = i - num_cols/2;  
    }

    global_result.resize(num_rows, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<TestTaskMPI>(taskDataPar);

  ASSERT_TRUE(taskParallel->validation());
  ASSERT_TRUE(taskParallel->pre_processing());
  ASSERT_TRUE(taskParallel->run());
  ASSERT_TRUE(taskParallel->post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_result(num_rows);
    for (int i = 0; i < num_rows; ++i) {
      int sum = 0;
      for (int j = 0; j < num_cols; ++j) {
        sum += global_matrix[i * num_cols + j] * global_vector[j];
      }
      expected_result[i] = sum;
    }

    ASSERT_EQ(global_result.size(), expected_result.size());
    for (size_t i = 0; i < global_result.size(); ++i) {
      ASSERT_EQ(global_result[i], expected_result[i]);
    }
  }
}
