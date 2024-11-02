#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <climits>
#include <random>
#include <vector>

#include "mpi/borisov_s_sum_of_rows/include/ops_mpi.hpp"

std::vector<int> getRandomMatrix(size_t rows, size_t cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> matrix(rows * cols);
  for (auto& element : matrix) {
    element = static_cast<int>(gen() % 100);
  }
  return matrix;
}

TEST(borisov_s_sum_of_rows, Test_Unit_Matrix) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_row_sums;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  size_t rows = 10;
  size_t cols = 10;

  if (world.rank() == 0) {
    global_matrix.resize(rows * cols, 1);
    global_row_sums.resize(rows, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_row_sums.data()));
    taskDataPar->outputs_count.push_back(global_row_sums.size());
  } else {
    taskDataPar->inputs.emplace_back(nullptr);
    taskDataPar->outputs.emplace_back(nullptr);
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs_count.push_back(0);
  }

  borisov_s_sum_of_rows::SumOfRowsTaskParallel sumOfRowsTaskParallel(taskDataPar);
  ASSERT_EQ(sumOfRowsTaskParallel.validation(), true);

  sumOfRowsTaskParallel.pre_processing();
  sumOfRowsTaskParallel.run();
  sumOfRowsTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (size_t i = 0; i < global_row_sums.size(); i++) {
      ASSERT_EQ(global_row_sums[i], 10);
    }
  }
}

TEST(borisov_s_sum_of_rows, Test_Zero_Matrix) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_row_sums;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  size_t rows = 10;
  size_t cols = 10;

  if (world.rank() == 0) {
    global_matrix.resize(rows * cols, 0);
    global_row_sums.resize(rows, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_row_sums.data()));
    taskDataPar->outputs_count.push_back(global_row_sums.size());
  } else {
    taskDataPar->inputs.emplace_back(nullptr);
    taskDataPar->outputs.emplace_back(nullptr);
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs_count.push_back(0);
  }

  borisov_s_sum_of_rows::SumOfRowsTaskParallel sumOfRowsTaskParallel(taskDataPar);
  ASSERT_EQ(sumOfRowsTaskParallel.validation(), true);

  sumOfRowsTaskParallel.pre_processing();
  sumOfRowsTaskParallel.run();
  sumOfRowsTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (size_t i = 0; i < global_row_sums.size(); i++) {
      ASSERT_EQ(global_row_sums[i], 0);
    }
  }
}

TEST(borisov_s_sum_of_rows, Test_Sum_Rows) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_row_sums;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  size_t rows = 15;
  size_t cols = 15;

  if (world.rank() == 0) {
    global_matrix = getRandomMatrix(rows, cols);
    global_row_sums.resize(rows, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_row_sums.data()));
    taskDataPar->outputs_count.push_back(global_row_sums.size());
  } else {
    taskDataPar->inputs.emplace_back(nullptr);
    taskDataPar->outputs.emplace_back(nullptr);
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs_count.push_back(0);
  }

  borisov_s_sum_of_rows::SumOfRowsTaskParallel sumOfRowsTaskParallel(taskDataPar);
  ASSERT_EQ(sumOfRowsTaskParallel.validation(), true);

  sumOfRowsTaskParallel.pre_processing();
  sumOfRowsTaskParallel.run();
  sumOfRowsTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_row_sums(global_row_sums.size(), 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.push_back(rows);
    taskDataSeq->inputs_count.push_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_row_sums.data()));
    taskDataSeq->outputs_count.push_back(reference_row_sums.size());

    borisov_s_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTaskSequential(taskDataSeq);
    ASSERT_EQ(sumOfRowsTaskSequential.validation(), true);
    sumOfRowsTaskSequential.pre_processing();
    sumOfRowsTaskSequential.run();
    sumOfRowsTaskSequential.post_processing();

    for (size_t i = 0; i < global_row_sums.size(); i++) {
      ASSERT_EQ(reference_row_sums[i], global_row_sums[i]);
    }
  }
}

TEST(borisov_s_sum_of_rows, Test_Empty_Matrix) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_row_sums;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  size_t rows = 0;
  size_t cols = 0;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_row_sums.data()));
    global_row_sums.resize(rows, 0);
  } else {
    taskDataPar->inputs.emplace_back(nullptr);
    taskDataPar->outputs.emplace_back(nullptr);
  }

  taskDataPar->inputs_count.push_back(rows);
  taskDataPar->inputs_count.push_back(cols);
  taskDataPar->outputs_count.push_back(global_row_sums.size());

  borisov_s_sum_of_rows::SumOfRowsTaskParallel sumOfRowsTaskParallel(taskDataPar);

  ASSERT_FALSE(sumOfRowsTaskParallel.validation());
}

TEST(borisov_s_sum_of_rows, Test_NonDivisibleRows) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_row_sums;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  size_t rows = 17;
  size_t cols = 10;

  if (world.rank() == 0) {
    global_matrix = getRandomMatrix(rows, cols);
    global_row_sums.resize(rows, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_row_sums.data()));
    taskDataPar->outputs_count.push_back(global_row_sums.size());
  } else {
    taskDataPar->inputs.emplace_back(nullptr);
    taskDataPar->outputs.emplace_back(nullptr);
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs_count.push_back(0);
  }

  borisov_s_sum_of_rows::SumOfRowsTaskParallel sumOfRowsTaskParallel(taskDataPar);
  ASSERT_EQ(sumOfRowsTaskParallel.validation(), true);

  sumOfRowsTaskParallel.pre_processing();
  sumOfRowsTaskParallel.run();
  sumOfRowsTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_row_sums(global_row_sums.size(), 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.push_back(rows);
    taskDataSeq->inputs_count.push_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_row_sums.data()));
    taskDataSeq->outputs_count.push_back(reference_row_sums.size());

    borisov_s_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTaskSequential(taskDataSeq);
    ASSERT_EQ(sumOfRowsTaskSequential.validation(), true);
    sumOfRowsTaskSequential.pre_processing();
    sumOfRowsTaskSequential.run();
    sumOfRowsTaskSequential.post_processing();

    for (size_t i = 0; i < global_row_sums.size(); i++) {
      ASSERT_EQ(reference_row_sums[i], global_row_sums[i]);
    }
  }
}

TEST(borisov_s_sum_of_rows, Test_Large_Matrix) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_row_sums;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  size_t rows = 10000;
  size_t cols = 1000;

  if (world.rank() == 0) {
    global_matrix = getRandomMatrix(rows, cols);
    global_row_sums.resize(rows, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_row_sums.data()));
    taskDataPar->outputs_count.push_back(global_row_sums.size());
  } else {
    taskDataPar->inputs.emplace_back(nullptr);
    taskDataPar->outputs.emplace_back(nullptr);
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs_count.push_back(0);
  }

  borisov_s_sum_of_rows::SumOfRowsTaskParallel sumOfRowsTaskParallel(taskDataPar);
  ASSERT_EQ(sumOfRowsTaskParallel.validation(), true);

  sumOfRowsTaskParallel.pre_processing();
  sumOfRowsTaskParallel.run();
  sumOfRowsTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (size_t i = 0; i < global_row_sums.size(); i++) {
      ASSERT_GE(global_row_sums[i], 0);
    }
  }
}

TEST(borisov_s_sum_of_rows, Test_Max_Min_Int) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_row_sums;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  size_t rows = 2;
  size_t cols = 2;

  if (world.rank() == 0) {
    global_matrix = {INT_MAX, INT_MIN, INT_MAX, INT_MIN};
    global_row_sums.resize(rows, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_row_sums.data()));
    taskDataPar->outputs_count.push_back(global_row_sums.size());
  } else {
    taskDataPar->inputs.emplace_back(nullptr);
    taskDataPar->outputs.emplace_back(nullptr);
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs_count.push_back(0);
  }

  borisov_s_sum_of_rows::SumOfRowsTaskParallel sumOfRowsTaskParallel(taskDataPar);
  ASSERT_EQ(sumOfRowsTaskParallel.validation(), true);

  sumOfRowsTaskParallel.pre_processing();
  sumOfRowsTaskParallel.run();
  sumOfRowsTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_row_sums = {INT_MAX + INT_MIN, INT_MAX + INT_MIN};
    for (size_t i = 0; i < rows; i++) {
      ASSERT_EQ(reference_row_sums[i], global_row_sums[i]);
    }
  }
}

TEST(borisov_s_sum_of_rows, Test_Same_Numbers_In_Row) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_row_sums;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  size_t rows = 5;
  size_t cols = 5;

  if (world.rank() == 0) {
    global_matrix.resize(rows * cols);
    for (size_t i = 0; i < rows; ++i) {
      int value = static_cast<int>(i + 1);
      for (size_t j = 0; j < cols; ++j) {
        global_matrix[(i * cols) + j] = value;
      }
    }
    global_row_sums.resize(rows, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_row_sums.data()));
    taskDataPar->outputs_count.push_back(global_row_sums.size());
  } else {
    taskDataPar->inputs.emplace_back(nullptr);
    taskDataPar->outputs.emplace_back(nullptr);
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs_count.push_back(0);
  }

  borisov_s_sum_of_rows::SumOfRowsTaskParallel sumOfRowsTaskParallel(taskDataPar);
  ASSERT_EQ(sumOfRowsTaskParallel.validation(), true);

  sumOfRowsTaskParallel.pre_processing();
  sumOfRowsTaskParallel.run();
  sumOfRowsTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (size_t i = 0; i < rows; i++) {
      int expected_sum = static_cast<int>((i + 1) * cols);
      ASSERT_EQ(global_row_sums[i], expected_sum);
    }
  }
}

TEST(borisov_s_sum_of_rows, Test_Null_Pointers) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  size_t rows = 10;
  size_t cols = 10;

  taskDataPar->inputs.emplace_back(nullptr);
  taskDataPar->outputs.emplace_back(nullptr);
  taskDataPar->inputs_count.push_back(rows);
  taskDataPar->inputs_count.push_back(cols);
  taskDataPar->outputs_count.push_back(rows);

  borisov_s_sum_of_rows::SumOfRowsTaskParallel sumOfRowsTaskParallel(taskDataPar);
  ASSERT_FALSE(sumOfRowsTaskParallel.validation());
}

TEST(borisov_s_sum_of_rows, Test_Invalid_Output_Count_Sequential) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<int> global_matrix;
  std::vector<int> global_row_sums;

  size_t rows = 10;
  size_t cols = 10;

  global_matrix.resize(rows * cols, 1);
  global_row_sums.resize(rows - 1, 0);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_row_sums.data()));
  taskDataSeq->outputs_count.push_back(global_row_sums.size());

  borisov_s_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTaskSequential(taskDataSeq);

  ASSERT_FALSE(sumOfRowsTaskSequential.validation());
}

TEST(borisov_s_sum_of_rows, Test_Invalid_Input_Output_Size_Sequential) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  size_t rows = 10;

  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->outputs_count.push_back(0);

  borisov_s_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTaskSequential(taskDataSeq);

  ASSERT_FALSE(sumOfRowsTaskSequential.validation());
}

TEST(borisov_s_sum_of_rows, Test_Validation_Invalid_Output_Count_Sequential) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<int> global_matrix;
  std::vector<int> global_row_sums;

  size_t rows = 10;
  size_t cols = 10;

  global_matrix.resize(rows * cols, 1);
  global_row_sums.resize(rows - 1, 0);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_row_sums.data()));
  taskDataSeq->outputs_count.push_back(global_row_sums.size());

  borisov_s_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTaskSequential(taskDataSeq);

  ASSERT_FALSE(sumOfRowsTaskSequential.validation());
}

TEST(borisov_s_sum_of_rows, Test_Validation_Cols_Less_Than_Or_Equal_To_Zero) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  size_t rows = 10;
  size_t cols = 0;

  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs_count.push_back(rows);

  borisov_s_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTaskSequential(taskDataSeq);

  ASSERT_FALSE(sumOfRowsTaskSequential.validation());
}

TEST(borisov_s_sum_of_rows, Test_Validation_Null_Pointers) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  size_t rows = 10;
  size_t cols = 10;

  taskDataSeq->inputs.emplace_back(nullptr);
  taskDataSeq->outputs.emplace_back(nullptr);
  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs_count.push_back(rows);

  borisov_s_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTaskSequential(taskDataSeq);

  ASSERT_FALSE(sumOfRowsTaskSequential.validation());
}

TEST(borisov_s_sum_of_rows, Test_Run_Empty_Matrix_Sequential) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<int> global_matrix;
  std::vector<int> global_row_sums;

  size_t rows = 0;
  size_t cols = 0;

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_row_sums.data()));
  taskDataSeq->outputs_count.push_back(global_row_sums.size());

  borisov_s_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTaskSequential(taskDataSeq);

  ASSERT_FALSE(sumOfRowsTaskSequential.validation());
  ASSERT_TRUE(sumOfRowsTaskSequential.pre_processing());
  ASSERT_TRUE(sumOfRowsTaskSequential.run());

  ASSERT_TRUE(sumOfRowsTaskSequential.post_processing());
}

TEST(borisov_s_sum_of_rows, Test_Run_NonEmpty_Matrix_Sequential) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<int> global_matrix;
  std::vector<int> global_row_sums;

  size_t rows = 3;
  size_t cols = 3;

  global_matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  global_row_sums.resize(rows, 0);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_row_sums.data()));
  taskDataSeq->outputs_count.push_back(global_row_sums.size());

  borisov_s_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTaskSequential(taskDataSeq);

  ASSERT_TRUE(sumOfRowsTaskSequential.validation());
  ASSERT_TRUE(sumOfRowsTaskSequential.pre_processing());
  ASSERT_TRUE(sumOfRowsTaskSequential.run());

  ASSERT_TRUE(sumOfRowsTaskSequential.post_processing());

  std::vector<int> expected_sums = {6, 15, 24};
  for (size_t i = 0; i < rows; i++) {
    ASSERT_EQ(global_row_sums[i], expected_sums[i]);
  }
}