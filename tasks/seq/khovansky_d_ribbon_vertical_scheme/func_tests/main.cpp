// Copyright 2024 Khovansky Dmitry
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/khovansky_d_ribbon_vertical_scheme/include/ops_seq.hpp"

TEST(khovansky_d_ribbon_vertical_scheme_seq, empty_matrix) {
  std::vector<int> matrix = {};
  std::vector<int> vector = {1, 2, 3};
  std::vector<int> result(3, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  khovansky_d_ribbon_vertical_scheme_seq::RibbonVerticalSchemeSeq ribbonVerticalSchemeSeq(taskDataSeq);
  ASSERT_EQ(ribbonVerticalSchemeSeq.validation(), false);
}

TEST(khovansky_d_ribbon_vertical_scheme_seq, empty_vector) {
  std::vector<int> matrix = {1, 2, 3, 4};
  std::vector<int> vector = {};
  std::vector<int> result(2, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  khovansky_d_ribbon_vertical_scheme_seq::RibbonVerticalSchemeSeq ribbonVerticalSchemeSeq(taskDataSeq);
  ASSERT_EQ(ribbonVerticalSchemeSeq.validation(), false);
}

TEST(khovansky_d_ribbon_vertical_scheme_seq, empty_matrix_and_vector) {
  std::vector<int> matrix = {};
  std::vector<int> vector = {};
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  khovansky_d_ribbon_vertical_scheme_seq::RibbonVerticalSchemeSeq ribbonVerticalSchemeSeq(taskDataSeq);
  ASSERT_EQ(ribbonVerticalSchemeSeq.validation(), false);
}

TEST(khovansky_d_ribbon_vertical_scheme_seq, 2x2_matrix_seq) {
  std::vector<int> matrix = {1, 2, 3, 4};
  std::vector<int> vector = {5, 6};
  std::vector<int> result(2, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  khovansky_d_ribbon_vertical_scheme_seq::RibbonVerticalSchemeSeq ribbonVerticalSchemeSeq(taskDataSeq);
  ASSERT_EQ(ribbonVerticalSchemeSeq.validation(), true);
  ribbonVerticalSchemeSeq.pre_processing();
  ribbonVerticalSchemeSeq.run();
  ribbonVerticalSchemeSeq.post_processing();

  std::vector<int> expected_result = {23, 34};
  ASSERT_EQ(result, expected_result);
}

TEST(khovansky_d_ribbon_vertical_scheme_seq, 3x3_matrix_seq) {
  std::vector<int> matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> vector = {1, 2, 3};
  std::vector<int> result(3, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  khovansky_d_ribbon_vertical_scheme_seq::RibbonVerticalSchemeSeq ribbonVerticalSchemeSeq(taskDataSeq);
  ASSERT_EQ(ribbonVerticalSchemeSeq.validation(), true);
  ribbonVerticalSchemeSeq.pre_processing();
  ribbonVerticalSchemeSeq.run();
  ribbonVerticalSchemeSeq.post_processing();

  std::vector<int> expected_result = {30, 36, 42};
  ASSERT_EQ(result, expected_result);
}

TEST(khovansky_d_ribbon_vertical_scheme_seq, 1x4_matrix_seq) {
  std::vector<int> matrix = {4, 3, 2, 1};
  std::vector<int> vector = {10};
  std::vector<int> result(4, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  khovansky_d_ribbon_vertical_scheme_seq::RibbonVerticalSchemeSeq ribbonVerticalSchemeSeq(taskDataSeq);
  ASSERT_EQ(ribbonVerticalSchemeSeq.validation(), true);
  ribbonVerticalSchemeSeq.pre_processing();
  ribbonVerticalSchemeSeq.run();
  ribbonVerticalSchemeSeq.post_processing();

  std::vector<int> expected_result = {40, 30, 20, 10};
  ASSERT_EQ(result, expected_result);
}

TEST(khovansky_d_ribbon_vertical_scheme_seq, 4x1_matrix_seq) {
  std::vector<int> matrix = {1, 2, 3, 4};
  std::vector<int> vector = {5, 6, 7, 8};
  std::vector<int> result(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  khovansky_d_ribbon_vertical_scheme_seq::RibbonVerticalSchemeSeq ribbonVerticalSchemeSeq(taskDataSeq);
  ASSERT_EQ(ribbonVerticalSchemeSeq.validation(), true);
  ribbonVerticalSchemeSeq.pre_processing();
  ribbonVerticalSchemeSeq.run();
  ribbonVerticalSchemeSeq.post_processing();

  std::vector<int> expected_result = {70};
  ASSERT_EQ(result, expected_result);
}
