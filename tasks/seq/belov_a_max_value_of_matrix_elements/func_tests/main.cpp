#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/belov_a_max_value_of_matrix_elements/include/ops_seq.hpp"

using namespace belov_a_max_value_of_matrix_elements_seq;

template <typename T = int>
std::vector<T> generate_random_matrix(int rows, int cols, const T& left = T{-1000}, const T& right = T{1000}) {
  std::vector<T> res(rows * cols);
  std::random_device dev;
  std::mt19937 gen(dev());
  for (size_t i = 0; i < res.size(); i++) {
    res[i] = left + static_cast<T>(gen() % int(right - left + 1));
  }
  return res;
}

TEST(belov_a_max_value_of_matrix_elements_seq, Test_Max_Value_Positive_Integers) {
  const int rows = 2;
  const int cols = 3;

  std::vector<int> matrix = {7, 24, 35, 5, 10, 13};
  std::vector<int> dimensions = {rows, cols};
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(dimensions.size());
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  MaxValueOfMatrixElementsSequential<int> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(out[0], 35);
}

TEST(belov_a_max_value_of_matrix_elements_seq, Test_Max_Value_Negative_Integers) {
  const int rows = 2;
  const int cols = 3;

  std::vector<int> matrix = {-7, -24, -3, -15, -10, -13};
  std::vector<int> dimensions = {rows, cols};
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(dimensions.size());
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  MaxValueOfMatrixElementsSequential<int> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(out[0], -3);
}

TEST(belov_a_max_value_of_matrix_elements_seq, Test_Max_Value_Mixed_Integers) {
  const int rows = 2;
  const int cols = 3;

  std::vector<int> matrix = {-7, 24, -3, 15, 0, -1};
  std::vector<int> dimensions = {rows, cols};
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(dimensions.size());
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  MaxValueOfMatrixElementsSequential<int> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(out[0], 24);
}

TEST(belov_a_max_value_of_matrix_elements_seq, Test_Max_Value_One_Element) {
  const int rows = 1;
  const int cols = 1;

  std::vector<int> matrix = {42};
  std::vector<int> dimensions = {rows, cols};
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(dimensions.size());
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  MaxValueOfMatrixElementsSequential<int> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(out[0], 42);
}

TEST(belov_a_max_value_of_matrix_elements_seq, Test_Max_Value_Positive_Doubles) {
  const int rows = 2;
  const int cols = 3;

  std::vector<double> matrix = {7.2, 24.1, 35.3, 5.5, 10.6, 13.9};
  std::vector<int> dimensions = {rows, cols};
  std::vector<double> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(dimensions.size());
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  MaxValueOfMatrixElementsSequential<double> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(out[0], 35.3);
}

TEST(belov_a_max_value_of_matrix_elements_seq, Test_Max_Value_Mixed_Doubles) {
  const int rows = 3;
  const int cols = 2;

  std::vector<double> matrix = {-10.1, 24.1, -3.5, 15.7, -0.5, -1.2};
  std::vector<int> dimensions = {rows, cols};
  std::vector<double> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(dimensions.size());
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  MaxValueOfMatrixElementsSequential<double> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(out[0], 24.1);
}

TEST(belov_a_max_value_of_matrix_elements_seq, Test_Max_Value_Zeroes_Integers) {
  const int rows = 3;
  const int cols = 3;

  std::vector<int> matrix = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> dimensions = {rows, cols};
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(dimensions.size());
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  MaxValueOfMatrixElementsSequential<int> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(out[0], 0);
}

TEST(belov_a_max_value_of_matrix_elements_seq, Test_Max_Value_All_Same_Integers) {
  const int rows = 2;
  const int cols = 4;

  std::vector<int> matrix = {8, 8, 8, 8, 8, 8, 8, 8};
  std::vector<int> dimensions = {rows, cols};
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(dimensions.size());
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  MaxValueOfMatrixElementsSequential<int> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(out[0], 8);
}

TEST(belov_a_max_value_of_matrix_elements_seq, Test_Max_Value_Float_Negative) {
  const int rows = 2;
  const int cols = 2;

  std::vector<float> matrix = {-7.5f, -0.2f, -15.3f, -5.9f};
  std::vector<int> dimensions = {rows, cols};
  std::vector<float> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(dimensions.size());
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  MaxValueOfMatrixElementsSequential<float> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(out[0], -0.2f);
}

TEST(belov_a_max_value_of_matrix_elements_seq, Test_Max_Value_Mixed_Zero_Positive_Negative) {
  const int rows = 2;
  const int cols = 3;

  std::vector<int> matrix = {0, -20, 5, 10, -5, 0};
  std::vector<int> dimensions = {rows, cols};
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(dimensions.size());
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  MaxValueOfMatrixElementsSequential<int> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(out[0], 10);
}

TEST(belov_a_max_value_of_matrix_elements_seq, Test_Max_Value_Large_Integers) {
  const int rows = 2;
  const int cols = 3;

  std::vector<int> matrix = {2147483647, -2147483648, 1, 0, 100, 999};
  std::vector<int> dimensions = {rows, cols};
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(dimensions.size());
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  MaxValueOfMatrixElementsSequential<int> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(out[0], 2147483647);
}

TEST(belov_a_max_value_of_matrix_elements_seq, Test_Max_Value_Large_Matrix_Diverse_Double) {
  const int rows = 1000;
  const int cols = 1000;

  // Creating a matrix with a variety of values: large, small, positive, negative, fractional
  std::vector<double> matrix(rows * cols);
  for (int i = 0; i < rows * cols; ++i) {
    if (i % 5 == 0) {
      matrix[i] = static_cast<double>(i * 1.1);
    } else if (i % 5 == 1) {
      matrix[i] = static_cast<double>(-i * 1.2);
    } else if (i % 5 == 2) {
      matrix[i] = static_cast<double>(i * 0.001);
    } else if (i % 5 == 3) {
      matrix[i] = -0.5;
    } else {
      matrix[i] = 1999999.99;
    }
  }

  std::vector<int> dimensions = {rows, cols};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(dimensions.size());
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  belov_a_max_value_of_matrix_elements_seq::MaxValueOfMatrixElementsSequential<double> testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(out[0], 1999999.99);
}

TEST(belov_a_max_value_of_matrix_elements_seq, Test_Validation_EmptyData) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  MaxValueOfMatrixElementsSequential<int> testTaskSequential(taskDataSeq);

  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(belov_a_max_value_of_matrix_elements_seq, Test_PreProcessing_NonPositiveDimensions) {
  std::vector<int> dimensions = {0, 5};
  std::vector<int> matrix = {};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(dimensions.size());
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  belov_a_max_value_of_matrix_elements_seq::MaxValueOfMatrixElementsSequential<double> testTaskSequential(taskDataSeq);

  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(belov_a_max_value_of_matrix_elements_seq, Test_Random_Matrix_Integers) {
  std::vector<int> matrix = generate_random_matrix<int>(10, 10);
  bool flag = true;

  for (const auto& item : matrix) {
    if (item < -1000 || item > 1000) {
      flag = false;
      break;
    }
  }

  ASSERT_TRUE(flag && matrix.size() == 100);
}