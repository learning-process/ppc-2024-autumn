#include <gtest/gtest.h>

#include <vector>

#include "seq/gromov_a_gaussian_method_vertical/include/ops_seq.hpp"

TEST(gromov_a_gaussian_method_vertical_seq, Test_Matrix_1) {
  std::vector<int> input_coefficient = {2, 3, 4, -1};
  std::vector<int> input_rhs = {8, 2};
  std::vector<double> func_res(input_rhs.size(), 0);
  std::vector<double> ans = {1, 2};
  int band_width = 3;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coefficient.data()));
  taskDataSeq->inputs_count.emplace_back(input_coefficient.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_rhs.data()));
  taskDataSeq->inputs_count.emplace_back(input_rhs.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
  taskDataSeq->outputs_count.emplace_back(func_res.size());

  gromov_a_gaussian_method_vertical_seq::GaussVerticalSequential gaussVerticalSequential(taskDataSeq, band_width);
  ASSERT_EQ(gaussVerticalSequential.validation(), true);
  gaussVerticalSequential.pre_processing();
  gaussVerticalSequential.run();
  gaussVerticalSequential.post_processing();
  ASSERT_EQ(ans, func_res);
}

TEST(gromov_a_gaussian_method_vertical_seq, Test_Matrix_2) {
  std::vector<int> input_coefficient = {1, 1, 2, -1};
  std::vector<int> input_rhs = {5, 4};
  std::vector<double> func_res(input_rhs.size(), 0);
  std::vector<double> ans = {3, 2};
  int band_width = 3;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coefficient.data()));
  taskDataSeq->inputs_count.emplace_back(input_coefficient.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_rhs.data()));
  taskDataSeq->inputs_count.emplace_back(input_rhs.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
  taskDataSeq->outputs_count.emplace_back(func_res.size());

  gromov_a_gaussian_method_vertical_seq::GaussVerticalSequential gaussVerticalSequential(taskDataSeq, band_width);
  ASSERT_EQ(gaussVerticalSequential.validation(), true);
  gaussVerticalSequential.pre_processing();
  gaussVerticalSequential.run();
  gaussVerticalSequential.post_processing();
  ASSERT_EQ(ans, func_res);
}

TEST(gromov_a_gaussian_method_vertical_seq, Test_Negative_Positive_Coefficients) {
  std::vector<int> input_coefficient = {-1, 2, 4, -5};
  std::vector<int> input_rhs = {3, -2};
  std::vector<double> func_res(input_rhs.size(), 0);
  std::vector<double> ans = {11.0 / 3.0, 10.0 / 3.0};
  int band_width = 2;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coefficient.data()));
  taskDataSeq->inputs_count.emplace_back(input_coefficient.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_rhs.data()));
  taskDataSeq->inputs_count.emplace_back(input_rhs.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
  taskDataSeq->outputs_count.emplace_back(func_res.size());

  gromov_a_gaussian_method_vertical_seq::GaussVerticalSequential gaussVerticalSequential(taskDataSeq, band_width);
  ASSERT_EQ(gaussVerticalSequential.validation(), true);
  gaussVerticalSequential.pre_processing();
  gaussVerticalSequential.run();
  gaussVerticalSequential.post_processing();

  double tolerance = 1e-6;
  for (size_t i = 0; i < ans.size(); ++i) {
    ASSERT_NEAR(ans[i], func_res[i], tolerance);
  }
}

TEST(gromov_a_gaussian_method_vertical_seq, Test_Band_Matrix) {
  std::vector<int> input_coefficient = {4, 1, 0, 0, 1, 4, 1, 0, 0, 1, 4, 1, 0, 0, 1, 4};
  std::vector<int> input_rhs = {5, 6, 6, 5};
  std::vector<double> func_res(input_rhs.size(), 0);
  std::vector<double> ans = {1, 1, 1, 1};
  int band_width = 2;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coefficient.data()));
  taskDataSeq->inputs_count.emplace_back(input_coefficient.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_rhs.data()));
  taskDataSeq->inputs_count.emplace_back(input_rhs.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
  taskDataSeq->outputs_count.emplace_back(func_res.size());

  gromov_a_gaussian_method_vertical_seq::GaussVerticalSequential gaussVerticalSequential(taskDataSeq, band_width);
  ASSERT_EQ(gaussVerticalSequential.validation(), true);
  gaussVerticalSequential.pre_processing();
  gaussVerticalSequential.run();
  gaussVerticalSequential.post_processing();
  ASSERT_EQ(ans, func_res);
}

TEST(gromov_a_gaussian_method_vertical_seq, Test_Matrix_3) {
  std::vector<int> input_coefficient = {1, 0, 0, 1};
  std::vector<int> input_rhs = {1, 1};
  std::vector<double> func_res(input_rhs.size(), 0);
  std::vector<double> ans = {1, 1};
  int band_width = 2;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coefficient.data()));
  taskDataSeq->inputs_count.emplace_back(input_coefficient.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_rhs.data()));
  taskDataSeq->inputs_count.emplace_back(input_rhs.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
  taskDataSeq->outputs_count.emplace_back(func_res.size());

  gromov_a_gaussian_method_vertical_seq::GaussVerticalSequential gaussVerticalSequential(taskDataSeq, band_width);
  ASSERT_EQ(gaussVerticalSequential.validation(), true);
  gaussVerticalSequential.pre_processing();
  gaussVerticalSequential.run();
  gaussVerticalSequential.post_processing();
  ASSERT_EQ(ans, func_res);
}

TEST(gromov_a_gaussian_method_vertical_seq, Test_Matrix_4) {
  std::vector<int> input_coefficient = {5};
  std::vector<int> input_rhs = {10};
  std::vector<double> func_res(input_rhs.size(), 0);
  std::vector<double> ans = {2};
  int band_width = 1;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coefficient.data()));
  taskDataSeq->inputs_count.emplace_back(input_coefficient.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_rhs.data()));
  taskDataSeq->inputs_count.emplace_back(input_rhs.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
  taskDataSeq->outputs_count.emplace_back(func_res.size());

  gromov_a_gaussian_method_vertical_seq::GaussVerticalSequential gaussVerticalSequential(taskDataSeq, band_width);
  ASSERT_EQ(gaussVerticalSequential.validation(), true);
  gaussVerticalSequential.pre_processing();
  gaussVerticalSequential.run();
  gaussVerticalSequential.post_processing();
  ASSERT_EQ(ans, func_res);
}