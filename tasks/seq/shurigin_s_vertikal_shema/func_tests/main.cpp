#include <gtest/gtest.h>

#include <vector>

#include "seq/shurigin_s_vertikal_shema/include/ops_seq.hpp"

TEST(shurigin_s_vertikal_shema, fail__validation) {
  std::vector<int> input_data = {42};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(1);

  shurigin_s_vertikal_shema::TestTaskSequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), false);
}

TEST(shurigin_s_vertikal_shema, empty_input_matrix) {
  std::vector<std::vector<int>> input_matrix = {};
  std::vector<int> input_vector = {1, 2};
  std::vector<int> output_result(0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(0);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  shurigin_s_vertikal_shema::TestTaskSequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), false);
}

TEST(shurigin_s_vertikal_shema, matrix_normal) {
  std::vector<int> matrix_data = {1, 2, 3, 4, 5, 6};  // Столбцы: {1,2,3}, {4,5,6}
  std::vector<int> input_vector = {7, 8};             // Вектор: {7, 8}
  std::vector<int> output_result(3, 0);               // Результат размера 3

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_data.data()));
  taskDataSeq->inputs_count.emplace_back(matrix_data.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  shurigin_s_vertikal_shema::TestTaskSequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), true);
  taskSequential.pre_processing();
  taskSequential.run();
  taskSequential.post_processing();

  std::vector<int> expected_result = {39, 54, 69};
  ASSERT_EQ(output_result, expected_result);
}

TEST(shurigin_s_vertikal_shema, minus_data_num) {
  std::vector<int> matrix_data = {-1, -2, -3, -4};  // Столбцы: {-1,-2}, {-3,-4}
  std::vector<int> input_vector = {-5, -6};         // Вектор: {-5, -6}
  std::vector<int> output_result(2, 0);             // Результат размера 2

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_data.data()));
  taskDataSeq->inputs_count.emplace_back(matrix_data.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  shurigin_s_vertikal_shema::TestTaskSequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), true);
  taskSequential.pre_processing();
  taskSequential.run();
  taskSequential.post_processing();

  std::vector<int> expected_result = {23, 34};
  ASSERT_EQ(output_result, expected_result);
}

TEST(shurigin_s_vertikal_shema, one_row) {
  std::vector<int> matrix_data = {1, 2, 3};
  std::vector<int> input_vector = {4, 5, 6};  // Вектор: {4, 5, 6}
  std::vector<int> output_result(1, 0);       // Результат размера 1

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_data.data()));
  taskDataSeq->inputs_count.emplace_back(matrix_data.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  shurigin_s_vertikal_shema::TestTaskSequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), true);
  taskSequential.pre_processing();
  taskSequential.run();
  taskSequential.post_processing();

  std::vector<int> expected_result = {32};
  ASSERT_EQ(output_result, expected_result);
}

TEST(shurigin_s_vertikal_shema, one_col) {
  std::vector<int> matrix_data = {1, 2, 3};
  std::vector<int> input_vector = {4};   // Вектор: {4}
  std::vector<int> output_result(3, 0);  // Результат размера 3

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_data.data()));
  taskDataSeq->inputs_count.emplace_back(matrix_data.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  shurigin_s_vertikal_shema::TestTaskSequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), true);
  taskSequential.pre_processing();
  taskSequential.run();
  taskSequential.post_processing();

  std::vector<int> expected_result = {4, 8, 12};
  ASSERT_EQ(output_result, expected_result);
}
