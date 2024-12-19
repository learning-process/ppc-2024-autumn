
#include <gtest/gtest.h>

#include <vector>

#include "seq/shurigin_lin_filtr_razbien_bloch_gaus_3x3/include/ops_seq.hpp"

TEST(shurigin_lin_filtr_razbien_bloch_gaus_3x3_seq, five_by_seven_matrix_with_linear_values) {
  const int num_rows = 5;
  const int num_cols = 7;
  std::vector<double> input_matrix = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                      25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0};
  std::vector<double> output_result(num_rows * num_cols, 0);
  std::vector<double> expected_result = {0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  9.0, 10.0, 11.0, 12.0,
                                         13.0, 0.0,  0.0,  16.0, 17.0, 18.0, 19.0, 20.0, 0.0, 0.0,  23.0, 24.0,
                                         25.0, 26.0, 27.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0,  0.0};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input_matrix.data())));
  task_data->inputs_count.push_back(input_matrix.size());

  int rows = num_rows;
  int cols = num_cols;
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));
  task_data->inputs_count.push_back(1);
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));
  task_data->inputs_count.push_back(1);

  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_result.data()));
  task_data->outputs_count.push_back(output_result.size());

  auto task = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3::TaskSeq>(task_data);
  ASSERT_TRUE(task->validation());
  task->pre_processing();
  task->run();
  task->post_processing();

  ASSERT_EQ(output_result, expected_result);
}

TEST(shurigin_lin_filtr_razbien_bloch_gaus_3x3_seq, five_by_seven_matrix_with_gradient) {
  const int num_rows = 5;
  const int num_cols = 7;
  std::vector<double> input_matrix = {1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0,
                                      3.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 2.0, 3.0, 4.0,
                                      5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0};
  std::vector<double> output_result(num_rows * num_cols, 0);
  std::vector<double> expected_result = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 4.5, 4.0,
                                         3.0, 0.0, 0.0, 3.5, 4.5, 5.0, 4.5, 3.5, 0.0, 0.0, 3.0, 4.0,
                                         4.5, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input_matrix.data())));
  task_data->inputs_count.push_back(input_matrix.size());

  int rows = num_rows;
  int cols = num_cols;
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));
  task_data->inputs_count.push_back(1);
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));
  task_data->inputs_count.push_back(1);

  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_result.data()));
  task_data->outputs_count.push_back(output_result.size());

  auto task = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3::TaskSeq>(task_data);
  ASSERT_TRUE(task->validation());
  task->pre_processing();
  task->run();
  task->post_processing();

  ASSERT_EQ(output_result, expected_result);
}

TEST(shurigin_lin_filtr_razbien_bloch_gaus_3x3_seq, validation_insufficient_rows) {
  const int num_rows = 2;
  const int num_cols = 7;
  std::vector<double> input_matrix = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  std::vector<double> output_result(num_rows * num_cols, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input_matrix.data())));
  task_data->inputs_count.push_back(input_matrix.size());

  int rows = num_rows;
  int cols = num_cols;
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));
  task_data->inputs_count.push_back(1);
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));
  task_data->inputs_count.push_back(1);

  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_result.data()));
  task_data->outputs_count.push_back(output_result.size());

  auto task = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3::TaskSeq>(task_data);
  ASSERT_FALSE(task->validation());
}

TEST(shurigin_lin_filtr_razbien_bloch_gaus_3x3_seq, validation_insufficient_columns) {
  const int num_rows = 5;
  const int num_cols = 2;
  std::vector<double> input_matrix = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  std::vector<double> output_result(num_rows * num_cols, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input_matrix.data())));
  task_data->inputs_count.push_back(input_matrix.size());

  int rows = num_rows;
  int cols = num_cols;
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));
  task_data->inputs_count.push_back(1);
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));
  task_data->inputs_count.push_back(1);

  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_result.data()));
  task_data->outputs_count.push_back(output_result.size());

  auto task = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3::TaskSeq>(task_data);
  ASSERT_FALSE(task->validation());
}

TEST(shurigin_lin_filtr_razbien_bloch_gaus_3x3_seq, validation_input) {
  const int num_rows = 5;
  const int num_cols = 7;
  std::vector<double> input_matrix = {1.0};
  std::vector<double> output_result(num_rows * num_cols, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input_matrix.data())));
  task_data->inputs_count.push_back(input_matrix.size());

  int rows = num_rows;
  int cols = num_cols;
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));
  task_data->inputs_count.push_back(1);
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));
  task_data->inputs_count.push_back(1);

  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_result.data()));
  task_data->outputs_count.push_back(output_result.size());

  auto task = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3::TaskSeq>(task_data);
  ASSERT_FALSE(task->validation());
}