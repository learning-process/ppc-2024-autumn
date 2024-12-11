// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/vector.hpp>
#include <random>
#include <vector>

#include "mpi/ermilova_d_custom_reduce/include/ops_mpi.hpp"

template <typename T>
static std::vector<T> getRandom(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distrib(-100, 100);
  std::vector<T> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = static_cast<T>(distrib(gen));
  }
  return vec;
}

template <typename T>
static void Test_reduce(MPI_Datatype datatype, MPI_Op op) {
  int size = 100;
  boost::mpi::communicator world;
  std::vector<T> input_;
  if (world.rank() == 0) {
    input_ = getRandom<T>(size * world.size());
  }
  std::vector<T> recv_data(size);
  boost::mpi::scatter(world, input_.data(), recv_data.data(), size, 0);

  T local_result = T{};
  if (op == MPI_SUM) {
    local_result = std::accumulate(recv_data.begin(), recv_data.end(), T{});
  } else if (op == MPI_MIN) {
    local_result = *std::min_element(recv_data.begin(), recv_data.end());
  } else if (op == MPI_MAX) {
    local_result = *std::max_element(recv_data.begin(), recv_data.end());
  }
  T global_result = T{};
  if (op == MPI_SUM) {
    global_result = T(0);
  } else if (op == MPI_MIN) {
    global_result = std::numeric_limits<T>::max();
  } else {
    global_result = std::numeric_limits<T>::min();
  }

  ermilova_d_custom_reduce_mpi::CustomReduce(&local_result, &global_result, 1, datatype, op, 0, MPI_COMM_WORLD);

  if (world.rank() == 0) {
    T ref = T{};
    if (op == MPI_SUM) {
      ref = std::accumulate(input_.begin(), input_.end(), T{});
    } else if (op == MPI_MIN) {
      ref = *std::min_element(input_.begin(), input_.end());
    } else if (op == MPI_MAX) {
      ref = *std::max_element(input_.begin(), input_.end());
    }
    ASSERT_EQ(static_cast<double>(global_result), static_cast<double>(ref));
  }
}

static std::vector<std::vector<int>> getRandomMatrix(int rows, int cols) {
  if (rows <= 0 || cols <= 0) throw "Incorrect size";
  std::vector<std::vector<int>> vec(rows);
  for (int i = 0; i < rows; i++) {
    vec[i] = getRandom<int>(cols);
  }
  return vec;
}

TEST(ermilova_d_custom_reduce_mpi, Can_create_vector) {
  const int size_test = 10;

  EXPECT_NO_THROW(getRandom<int>(size_test));
}

TEST(ermilova_d_custom_reduce_mpi, Cant_create_incorrect_size_vector) {
  const int size_test = -10;

  EXPECT_ANY_THROW(getRandom<int>(size_test));
}

TEST(ermilova_d_custom_reduce_mpi, Can_create_matrix) {
  const int rows_test = 10;
  const int cols_test = 10;
  EXPECT_NO_THROW(getRandomMatrix(rows_test, cols_test));
}

TEST(ermilova_d_custom_reduce_mpi, Cant_create_incorrect_size_matrix) {
  const int rows_test = -10;
  const int cols_test = 0;

  EXPECT_ANY_THROW(getRandomMatrix(rows_test, cols_test));
}

TEST(ermilova_d_custom_reduce_mpi, CustomReduce_int_sum) { Test_reduce<int>(MPI_INT, MPI_SUM); }

TEST(ermilova_d_custom_reduce_mpi, CustomReduce_int_min) { Test_reduce<int>(MPI_INT, MPI_MIN); }

TEST(ermilova_d_custom_reduce_mpi, CustomReduce_int_max) { Test_reduce<int>(MPI_INT, MPI_MAX); }

TEST(ermilova_d_custom_reduce_mpi, CustomReduce_float_sum) { Test_reduce<float>(MPI_FLOAT, MPI_SUM); }

TEST(ermilova_d_custom_reduce_mpi, CustomReduce_float_min) { Test_reduce<float>(MPI_FLOAT, MPI_MIN); }

TEST(ermilova_d_custom_reduce_mpi, CustomReduce_float_max) { Test_reduce<float>(MPI_FLOAT, MPI_MAX); }

TEST(ermilova_d_custom_reduce_mpi, CustomReduce_double_sum) { Test_reduce<double>(MPI_DOUBLE, MPI_SUM); }

TEST(ermilova_d_custom_reduce_mpi, CustomReduce_double_min) { Test_reduce<double>(MPI_DOUBLE, MPI_MIN); }

TEST(ermilova_d_custom_reduce_mpi, CustomReduce_double_max) { Test_reduce<double>(MPI_DOUBLE, MPI_MAX); }

TEST(ermilova_d_custom_reduce_mpi, Matrix_1x1) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);

  const int rows_test = 1;
  const int cols_test = 1;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = getRandomMatrix(rows_test, cols_test);
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(rows_test);
    taskDataPar->inputs_count.emplace_back(cols_test);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  ermilova_d_custom_reduce_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    int reference_min = INT_MAX;
    for (auto& row : global_matrix) {
      int row_min = *std::min_element(row.begin(), row.end());
      reference_min = std::min(reference_min, row_min);
    }

    ASSERT_EQ(reference_min, global_min[0]);
  }
}

TEST(ermilova_d_custom_reduce_mpi, Matrix_10x10) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);
  const int rows_test = 10;
  const int cols_test = 10;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = getRandomMatrix(rows_test, cols_test);
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(rows_test);
    taskDataPar->inputs_count.emplace_back(cols_test);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  ermilova_d_custom_reduce_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    int reference_min = INT_MAX;
    for (auto& row : global_matrix) {
      int row_min = *std::min_element(row.begin(), row.end());
      reference_min = std::min(reference_min, row_min);
    }

    ASSERT_EQ(reference_min, global_min[0]);
  }
}

TEST(ermilova_d_custom_reduce_mpi, Matrix_100x100) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);
  const int rows_test = 100;
  const int cols_test = 100;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = getRandomMatrix(rows_test, cols_test);
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(rows_test);
    taskDataPar->inputs_count.emplace_back(cols_test);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  ermilova_d_custom_reduce_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    int reference_min = INT_MAX;
    for (auto& row : global_matrix) {
      int row_min = *std::min_element(row.begin(), row.end());
      reference_min = std::min(reference_min, row_min);
    }

    ASSERT_EQ(reference_min, global_min[0]);
  }
}

TEST(ermilova_d_custom_reduce_mpi, Matrix_100x50) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);
  const int rows_test = 100;
  const int cols_test = 50;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = getRandomMatrix(rows_test, cols_test);
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(rows_test);
    taskDataPar->inputs_count.emplace_back(cols_test);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  ermilova_d_custom_reduce_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    int reference_min = INT_MAX;
    for (auto& row : global_matrix) {
      int row_min = *std::min_element(row.begin(), row.end());
      reference_min = std::min(reference_min, row_min);
    }

    ASSERT_EQ(reference_min, global_min[0]);
  }
}

TEST(ermilova_d_custom_reduce_mpi, Matrix_50x100) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);
  const int rows_test = 50;
  const int cols_test = 100;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = getRandomMatrix(rows_test, cols_test);
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(rows_test);
    taskDataPar->inputs_count.emplace_back(cols_test);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  ermilova_d_custom_reduce_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    int reference_min = INT_MAX;
    for (auto& row : global_matrix) {
      int row_min = *std::min_element(row.begin(), row.end());
      reference_min = std::min(reference_min, row_min);
    }

    ASSERT_EQ(reference_min, global_min[0]);
  }
}

TEST(ermilova_d_custom_reduce_mpi, Matrix_500x500) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);
  const int rows_test = 500;
  const int cols_test = 500;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = getRandomMatrix(rows_test, cols_test);
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(rows_test);
    taskDataPar->inputs_count.emplace_back(cols_test);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  ermilova_d_custom_reduce_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    int reference_min = INT_MAX;
    for (auto& row : global_matrix) {
      int row_min = *std::min_element(row.begin(), row.end());
      reference_min = std::min(reference_min, row_min);
    }

    ASSERT_EQ(reference_min, global_min[0]);
  }
}
