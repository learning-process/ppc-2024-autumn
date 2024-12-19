#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/somov_i_horizontal_scheme/include/ops_mpi.hpp"

namespace somov_i_horizontal_scheme {

std::vector<int32_t> create_random_vector(uint32_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0f, 300.0f);

  std::vector<int32_t> vec(size);
  for (auto &el : vec) {
    el = std::clamp(static_cast<int32_t>(std::round(dist(gen))), -900, 900);
  }
  return vec;
}

std::vector<int32_t> create_random_matrix(uint32_t rowCount, uint32_t colCount) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0f, 300.0f);

  std::vector<int32_t> matrix(rowCount * colCount);
  for (auto &el : matrix) {
    el = std::clamp(static_cast<int32_t>(std::round(dist(gen))), -900, 900);
  }
  return matrix;
}

}  // namespace somov_i_horizontal_scheme

TEST(somov_i_horizontal_scheme, test_large_matrix_distribution_and_algorithm_correctness) {
  int total_rows = 10000;
  int total_cols = 10000;
  int num_processes = 4;
  std::vector<int32_t> chunk_sizes;
  std::vector<int32_t> offsets;

  somov_i_horizontal_scheme::distribute_matrix_rows(total_rows, total_cols, num_processes, chunk_sizes, offsets);

  ASSERT_EQ(static_cast<int>(chunk_sizes.size()), num_processes);
  ASSERT_EQ(static_cast<int>(offsets.size()), num_processes);

  int total_assigned_rows = 0;
  std::vector<bool> covered_rows(total_rows, false);

  for (int i = 0; i < num_processes; ++i) {
    int rows_assigned = chunk_sizes[i] / total_cols;
    int start_row = offsets[i] / total_cols;

    if (chunk_sizes[i] > 0) {
      for (int row = start_row; row < start_row + rows_assigned; ++row) {
        ASSERT_FALSE(covered_rows[row]);
        covered_rows[row] = true;
      }
    }

    total_assigned_rows += rows_assigned;
  }

  EXPECT_EQ(total_assigned_rows, total_rows);
  for (bool covered : covered_rows) {
    EXPECT_TRUE(covered);
  }
}

TEST(somov_i_horizontal_scheme, test_distribution_more_processes_than_rows) {
  int total_rows = 3;
  int total_cols = 4;
  int num_processes = 4;
  std::vector<int32_t> chunk_sizes;
  std::vector<int32_t> offsets;

  somov_i_horizontal_scheme::distribute_matrix_rows(total_rows, total_cols, num_processes, chunk_sizes, offsets);

  ASSERT_EQ(static_cast<int>(chunk_sizes.size()), num_processes);
  ASSERT_EQ(static_cast<int>(offsets.size()), num_processes);

  for (int idx = 0; idx < num_processes; ++idx) {
    if (idx < total_rows) {
      EXPECT_EQ(chunk_sizes[idx], total_cols);
      EXPECT_EQ(offsets[idx], idx * total_cols);
    } else {
      EXPECT_EQ(chunk_sizes[idx], 0);
      EXPECT_EQ(offsets[idx], 0);
    }
  }
}

TEST(somov_i_horizontal_scheme, test_distribution_more_rows_than_processes) {
  int total_rows = 6;
  int total_cols = 4;
  int num_processes = 4;
  std::vector<int32_t> chunk_sizes;
  std::vector<int32_t> offsets;

  somov_i_horizontal_scheme::distribute_matrix_rows(total_rows, total_cols, num_processes, chunk_sizes, offsets);

  ASSERT_EQ(static_cast<int>(chunk_sizes.size()), num_processes);
  ASSERT_EQ(static_cast<int>(offsets.size()), num_processes);

  for (int idx = 0; idx < num_processes; ++idx) {
    if (idx < total_rows % num_processes) {
      EXPECT_EQ(chunk_sizes[idx], total_cols * (total_rows / num_processes + 1));
      EXPECT_EQ(offsets[idx], idx * (total_rows / num_processes + 1) * total_cols);
    } else {
      EXPECT_EQ(chunk_sizes[idx], total_cols * (total_rows / num_processes));
      EXPECT_EQ(offsets[idx], (idx * (total_rows / num_processes) + total_rows % num_processes) * total_cols);
    }
  }
}

TEST(somov_i_horizontal_scheme, test_distribution_fewer_processes) {
  int total_rows = 16;
  int total_cols = 4;
  int num_processes = 4;
  std::vector<int32_t> chunk_sizes;
  std::vector<int32_t> offsets;

  somov_i_horizontal_scheme::distribute_matrix_rows(total_rows, total_cols, num_processes, chunk_sizes, offsets);

  ASSERT_EQ(static_cast<int>(chunk_sizes.size()), num_processes);
  ASSERT_EQ(static_cast<int>(offsets.size()), num_processes);

  const int expected_chunk_sizes[] = {16, 16, 16, 16};
  const int expected_offsets[] = {0, 16, 32, 48};

  for (int idx = 0; idx < num_processes; ++idx) {
    EXPECT_EQ(chunk_sizes[idx], expected_chunk_sizes[idx]);
    EXPECT_EQ(offsets[idx], expected_offsets[idx]);
  }
}

TEST(somov_i_horizontal_scheme, test_distribution_with_exact_division) {
  int total_rows = 12;
  int total_cols = 3;
  int num_processes = 4;
  std::vector<int32_t> chunk_sizes;
  std::vector<int32_t> offsets;

  somov_i_horizontal_scheme::distribute_matrix_rows(total_rows, total_cols, num_processes, chunk_sizes, offsets);

  ASSERT_EQ(static_cast<int>(chunk_sizes.size()), num_processes);
  ASSERT_EQ(static_cast<int>(offsets.size()), num_processes);

  const int expected_chunk_sizes[] = {9, 9, 9, 9};
  const int expected_offsets[] = {0, 9, 18, 27};

  for (int idx = 0; idx < num_processes; ++idx) {
    EXPECT_EQ(chunk_sizes[idx], expected_chunk_sizes[idx]);
    EXPECT_EQ(offsets[idx], expected_offsets[idx]);
  }
}

TEST(somov_i_horizontal_scheme, test_invalid_matrix_vector_size) {
  boost::mpi::communicator mpi_world;
  std::vector<int32_t> matrix_data = {1, 1, 1, 2, 2, 2, 1, 1, 1};
  std::vector<int32_t> input_vector = {1};
  std::vector<int32_t> output_vector;
  std::shared_ptr<ppc::core::TaskData> parallel_task_data = std::make_shared<ppc::core::TaskData>();

  if (mpi_world.rank() == 0) {
    output_vector.resize(3);

    parallel_task_data->inputs.push_back(reinterpret_cast<uint8_t *>(matrix_data.data()));
    parallel_task_data->inputs_count.push_back(matrix_data.size());
    parallel_task_data->inputs.push_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    parallel_task_data->inputs_count.push_back(input_vector.size());
    parallel_task_data->outputs.push_back(reinterpret_cast<uint8_t *>(output_vector.data()));
    parallel_task_data->outputs_count.push_back(output_vector.size());
  }

  auto parallel_task = std::make_shared<somov_i_horizontal_scheme::MatrixVectorTaskMPI>(parallel_task_data);
  EXPECT_TRUE(parallel_task->validation());
}

TEST(somov_i_horizontal_scheme, test_empty_matrix_case) {
  boost::mpi::communicator mpi_world;
  std::vector<int32_t> matrix_data;
  std::vector<int32_t> input_vector = {2, 2, 2};
  std::vector<int32_t> result_vector;
  std::shared_ptr<ppc::core::TaskData> task_data_parallel = std::make_shared<ppc::core::TaskData>();

  if (mpi_world.rank() == 0) {
    result_vector.resize(0);

    task_data_parallel->inputs.push_back(reinterpret_cast<uint8_t *>(matrix_data.data()));
    task_data_parallel->inputs_count.push_back(matrix_data.size());
    task_data_parallel->inputs.push_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    task_data_parallel->inputs_count.push_back(input_vector.size());
    task_data_parallel->outputs.push_back(reinterpret_cast<uint8_t *>(result_vector.data()));
    task_data_parallel->outputs_count.push_back(result_vector.size());
  }

  auto parallel_task = std::make_shared<somov_i_horizontal_scheme::MatrixVectorTaskMPI>(task_data_parallel);

  EXPECT_TRUE(parallel_task->validation());
}

TEST(somov_i_horizontal_scheme, test_empty_input_vector) {
  boost::mpi::communicator mpi_world;
  std::vector<int32_t> matrix_data = {3, 3, 3, 3};
  std::vector<int32_t> input_vector;
  std::vector<int32_t> output_vector;
  std::shared_ptr<ppc::core::TaskData> parallel_task_data = std::make_shared<ppc::core::TaskData>();

  if (mpi_world.rank() == 0) {
    output_vector.resize(3);

    parallel_task_data->inputs.push_back(reinterpret_cast<uint8_t *>(matrix_data.data()));
    parallel_task_data->inputs_count.push_back(matrix_data.size());
    parallel_task_data->inputs.push_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    parallel_task_data->inputs_count.push_back(input_vector.size());
    parallel_task_data->outputs.push_back(reinterpret_cast<uint8_t *>(output_vector.data()));
    parallel_task_data->outputs_count.push_back(output_vector.size());
  }

  auto parallel_task = std::make_shared<somov_i_horizontal_scheme::MatrixVectorTaskMPI>(parallel_task_data);
  EXPECT_TRUE(parallel_task->validation());
}

TEST(somov_i_horizontal_scheme, matrix_vector_test) {
  boost::mpi::communicator world;

  std::vector<int32_t> matrix_data;
  std::vector<int32_t> input_vector;
  std::vector<int32_t> result_vector;

  std::shared_ptr<ppc::core::TaskData> task_data_parallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix_data = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                   19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36};

    input_vector = {1, 2, 3, 4, 5, 6};

    result_vector.resize(matrix_data.size() / input_vector.size());

    task_data_parallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_data.data()));
    task_data_parallel->inputs_count.emplace_back(matrix_data.size());

    task_data_parallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    task_data_parallel->inputs_count.emplace_back(input_vector.size());

    task_data_parallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_vector.data()));
    task_data_parallel->outputs_count.emplace_back(result_vector.size());
  }

  auto task_parallel = std::make_shared<somov_i_horizontal_scheme::MatrixVectorTaskMPI>(task_data_parallel);
  task_parallel->validation();
  task_parallel->pre_processing();
  task_parallel->run();
  task_parallel->post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> seq_result(result_vector.size());

    auto task_data_sequential = std::make_shared<ppc::core::TaskData>();
    task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_data.data()));
    task_data_sequential->inputs_count.emplace_back(matrix_data.size());

    task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    task_data_sequential->inputs_count.emplace_back(input_vector.size());

    task_data_sequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_result.data()));
    task_data_sequential->outputs_count.emplace_back(seq_result.size());

    auto task_sequential = std::make_shared<somov_i_horizontal_scheme::MatrixVectorTask>(task_data_sequential);
    task_sequential->validation();
    task_sequential->pre_processing();
    task_sequential->run();
    task_sequential->post_processing();

    ASSERT_EQ(result_vector.size(), seq_result.size());
    for (size_t i = 0; i < result_vector.size(); ++i) {
      EXPECT_EQ(result_vector[i], seq_result[i]);
    }
  }
}

TEST(somov_i_horizontal_scheme, negative_matrix_vector_test) {
  boost::mpi::communicator world;

  std::vector<int32_t> matrix_data;
  std::vector<int32_t> input_vector;
  std::vector<int32_t> result_vector;

  std::shared_ptr<ppc::core::TaskData> task_data_parallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix_data = {-10, 12, 5, -6, 8, -2, 0, 7, 14, -1, -9, 3};

    input_vector = {-1, 2, -3};

    result_vector.resize(matrix_data.size() / input_vector.size());

    task_data_parallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_data.data()));
    task_data_parallel->inputs_count.emplace_back(matrix_data.size());

    task_data_parallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    task_data_parallel->inputs_count.emplace_back(input_vector.size());

    task_data_parallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_vector.data()));
    task_data_parallel->outputs_count.emplace_back(result_vector.size());
  }

  auto task_parallel = std::make_shared<somov_i_horizontal_scheme::MatrixVectorTaskMPI>(task_data_parallel);
  task_parallel->validation();
  task_parallel->pre_processing();
  task_parallel->run();
  task_parallel->post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> seq_result(result_vector.size());

    auto task_data_sequential = std::make_shared<ppc::core::TaskData>();
    task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_data.data()));
    task_data_sequential->inputs_count.emplace_back(matrix_data.size());

    task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    task_data_sequential->inputs_count.emplace_back(input_vector.size());

    task_data_sequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_result.data()));
    task_data_sequential->outputs_count.emplace_back(seq_result.size());

    auto task_sequential = std::make_shared<somov_i_horizontal_scheme::MatrixVectorTask>(task_data_sequential);
    task_sequential->validation();
    task_sequential->pre_processing();
    task_sequential->run();
    task_sequential->post_processing();

    ASSERT_EQ(result_vector.size(), seq_result.size());
    for (size_t i = 0; i < result_vector.size(); ++i) {
      EXPECT_EQ(result_vector[i], seq_result[i]);
    }
  }
}

TEST(somov_i_horizontal_scheme, zero_matrix_vector_test) {
  boost::mpi::communicator world;

  std::vector<int32_t> matrix_data;
  std::vector<int32_t> input_vector;
  std::vector<int32_t> result_vector;

  std::shared_ptr<ppc::core::TaskData> task_data_parallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix_data = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    input_vector = {0, 0, 0, 0, 0, 0};

    result_vector.resize(matrix_data.size() / input_vector.size());

    task_data_parallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_data.data()));
    task_data_parallel->inputs_count.emplace_back(matrix_data.size());

    task_data_parallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    task_data_parallel->inputs_count.emplace_back(input_vector.size());

    task_data_parallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_vector.data()));
    task_data_parallel->outputs_count.emplace_back(result_vector.size());
  }

  auto task_parallel = std::make_shared<somov_i_horizontal_scheme::MatrixVectorTaskMPI>(task_data_parallel);
  task_parallel->validation();
  task_parallel->pre_processing();
  task_parallel->run();
  task_parallel->post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> seq_result(result_vector.size());

    auto task_data_sequential = std::make_shared<ppc::core::TaskData>();
    task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_data.data()));
    task_data_sequential->inputs_count.emplace_back(matrix_data.size());

    task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    task_data_sequential->inputs_count.emplace_back(input_vector.size());

    task_data_sequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_result.data()));
    task_data_sequential->outputs_count.emplace_back(seq_result.size());

    auto task_sequential = std::make_shared<somov_i_horizontal_scheme::MatrixVectorTask>(task_data_sequential);
    task_sequential->validation();
    task_sequential->pre_processing();
    task_sequential->run();
    task_sequential->post_processing();

    ASSERT_EQ(result_vector.size(), seq_result.size());
    for (size_t i = 0; i < result_vector.size(); ++i) {
      EXPECT_EQ(result_vector[i], seq_result[i]);
    }
  }
}
