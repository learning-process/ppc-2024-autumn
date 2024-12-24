#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/vershinina_a_cannons_algorithm/include/ops_seq.hpp"

vershinina_a_cannons_algorithm::TMatrix<double> getRandomMatrix(double r) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distr(0, 100);
  auto matrix = vershinina_a_cannons_algorithm::TMatrix<double>::create(r);
  for (size_t i = 0; i < matrix.n * matrix.n; i++) {
    matrix.data[i] = distr(gen);
  }
  return matrix;
}

TEST(vershinina_a_cannons_algorithm, Test_1) {
  auto lhs = vershinina_a_cannons_algorithm::TMatrix<double>::create(
      4, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  auto rhs = vershinina_a_cannons_algorithm::TMatrix<double>::create(
      4, {1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16});

  auto act_res = vershinina_a_cannons_algorithm::TMatrix<double>::create(4);

  auto ref_res = vershinina_a_cannons_algorithm::TMatrix<double>::create(
      4, {30, 70, 110, 150, 70, 174, 278, 382, 110, 278, 446, 614, 150, 382, 614, 846});
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lhs.data.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhs.data.data()));
  taskDataSeq->inputs_count.emplace_back(4);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(act_res.data.data()));
  taskDataSeq->outputs_count.emplace_back(act_res.n);

  vershinina_a_cannons_algorithm::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(act_res, ref_res);
}

TEST(vershinina_a_cannons_algorithm, Test_2) {
  auto lhs =
      vershinina_a_cannons_algorithm::TMatrix<double>::create(4, {2, 3, 4, 5, 9, 8, 7, 6, 5, 4, 2, 3, 8, 7, 3, 4});
  auto rhs =
      vershinina_a_cannons_algorithm::TMatrix<double>::create(4, {3, 5, 7, 6, 2, 7, 6, 3, 7, 5, 3, 2, 4, 3, 2, 5});

  auto act_res = vershinina_a_cannons_algorithm::TMatrix<double>::create(4);

  auto ref_res = vershinina_a_cannons_algorithm::TMatrix<double>::create(
      4, {60, 66, 54, 54, 116, 154, 144, 122, 49, 72, 71, 61, 75, 116, 115, 95});
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lhs.data.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhs.data.data()));
  taskDataSeq->inputs_count.emplace_back(4);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(act_res.data.data()));
  taskDataSeq->outputs_count.emplace_back(act_res.n);

  vershinina_a_cannons_algorithm::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(act_res, ref_res);
}

TEST(vershinina_a_cannons_algorithm, Test_3) {
  auto lhs = vershinina_a_cannons_algorithm::TMatrix<double>::create(3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto rhs = vershinina_a_cannons_algorithm::TMatrix<double>::create(3, {1, 4, 7, 2, 5, 8, 3, 6, 9});

  auto act_res = vershinina_a_cannons_algorithm::TMatrix<double>::create(3);

  auto ref_res = vershinina_a_cannons_algorithm::TMatrix<double>::create(3, {14, 32, 50, 32, 77, 122, 50, 122, 194});
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lhs.data.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhs.data.data()));
  taskDataSeq->inputs_count.emplace_back(3);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(act_res.data.data()));
  taskDataSeq->outputs_count.emplace_back(act_res.n);

  vershinina_a_cannons_algorithm::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(act_res, ref_res);
}

TEST(vershinina_a_cannons_algorithm, Test_4) {
  auto lhs = getRandomMatrix(3);
  auto rhs = getRandomMatrix(3);

  auto res = vershinina_a_cannons_algorithm::TMatrix<double>::create(3);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lhs.data.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhs.data.data()));
  taskDataSeq->inputs_count.emplace_back(3);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data.data()));
  taskDataSeq->outputs_count.emplace_back(res.n);

  vershinina_a_cannons_algorithm::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(res, lhs * rhs);
}

TEST(vershinina_a_cannons_algorithm, Test_5) {
  auto lhs = getRandomMatrix(10);
  auto rhs = getRandomMatrix(10);

  auto res = vershinina_a_cannons_algorithm::TMatrix<double>::create(10);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lhs.data.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhs.data.data()));
  taskDataSeq->inputs_count.emplace_back(10);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data.data()));
  taskDataSeq->outputs_count.emplace_back(res.n);

  vershinina_a_cannons_algorithm::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(res, lhs * rhs);
}
