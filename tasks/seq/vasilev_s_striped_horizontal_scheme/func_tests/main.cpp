#include <gtest/gtest.h>

#include <vector>

#include "seq/vasilev_s_striped_horizontal_scheme/include/ops_seq.hpp"

TEST(vasilev_s_striped_horizontal_scheme_seq, Validation_Failure) {
  std::vector<int> input_data = {42};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(1);

  vasilev_s_striped_horizontal_scheme_seq::StripedHorizontalSchemeSequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), false);
}

TEST(vasilev_s_striped_horizontal_scheme_seq, Empty_Matrix) {
  std::vector<int> input_matrix = {};
  std::vector<int> input_vector = {1, 2};
  std::vector<int> output_result(0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  vasilev_s_striped_horizontal_scheme_seq::StripedHorizontalSchemeSequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), false);
}

TEST(vasilev_s_striped_horizontal_scheme_seq, Rectangular_Matrix) {
  std::vector<int> input_matrix = {1, 2, 3, 4, 5, 6};
  std::vector<int> input_vector = {7, 8};
  std::vector<int> output_result(3);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  vasilev_s_striped_horizontal_scheme_seq::StripedHorizontalSchemeSequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), true);
  taskSequential.pre_processing();
  taskSequential.run();
  taskSequential.post_processing();

  std::vector<int> expected_result = {23, 53, 83};
  ASSERT_EQ(output_result, expected_result);
}

TEST(vasilev_s_striped_horizontal_scheme_seq, Negative_Numbers) {
  std::vector<int> input_matrix = {-1, -2, -3, -4};
  std::vector<int> input_vector = {-5, -6};
  std::vector<int> output_result(2);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  vasilev_s_striped_horizontal_scheme_seq::StripedHorizontalSchemeSequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), true);
  taskSequential.pre_processing();
  taskSequential.run();
  taskSequential.post_processing();

  std::vector<int> expected_result = {17, 39};
  ASSERT_EQ(output_result, expected_result);
}

TEST(vasilev_s_striped_horizontal_scheme_seq, Single_Row_Matrix) {
  std::vector<int> input_matrix = {1, 2, 3};
  std::vector<int> input_vector = {4, 5, 6};
  std::vector<int> output_result(1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  vasilev_s_striped_horizontal_scheme_seq::StripedHorizontalSchemeSequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), true);
  taskSequential.pre_processing();
  taskSequential.run();
  taskSequential.post_processing();

  std::vector<int> expected_result = {32};
  ASSERT_EQ(output_result, expected_result);
}

TEST(vasilev_s_striped_horizontal_scheme_seq, Single_Column_Matrix) {
  std::vector<int> input_matrix = {1, 2, 3};
  std::vector<int> input_vector = {4};
  std::vector<int> output_result(3);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  vasilev_s_striped_horizontal_scheme_seq::StripedHorizontalSchemeSequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), true);
  taskSequential.pre_processing();
  taskSequential.run();
  taskSequential.post_processing();

  std::vector<int> expected_result = {4, 8, 12};
  ASSERT_EQ(output_result, expected_result);
}
