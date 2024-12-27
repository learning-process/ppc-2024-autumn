// Copyright 2024 Sdobnov Vladimir

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "seq/Sdobnov_V_mergesort_Betcher/include/ops_seq.hpp"

TEST(Sdobnov_V_mergesort_Betcher_seq, InvalidInputCount) {
  int size = 8;
  std::vector<int> res(size, 0);
  std::vector<int> input = {2, 1, 0, 3, 2, 6, 4, 8};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->outputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));

  Sdobnov_V_mergesort_Betcher_seq::MergesortBetcherSeq test(taskDataSeq);

  ASSERT_FALSE(test.validation());
}

TEST(Sdobnov_V_mergesort_Betcher_seq, InvalidInputCountNotPow2) {
  int size = 9;
  std::vector<int> res(size, 0);
  std::vector<int> input = {2, 1, 0, 3, 2, 6, 4, 8, 9};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->outputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));

  Sdobnov_V_mergesort_Betcher_seq::MergesortBetcherSeq test(taskDataSeq);

  ASSERT_FALSE(test.validation());
}

TEST(Sdobnov_V_mergesort_Betcher_seq, InvalidInput) {
  int size = 8;
  std::vector<int> res(size, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->outputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));

  Sdobnov_V_mergesort_Betcher_seq::MergesortBetcherSeq test(taskDataSeq);

  ASSERT_FALSE(test.validation());
}

TEST(Sdobnov_V_mergesort_Betcher_seq, InvalidOutputCount) {
  int size = 8;
  std::vector<int> res(size, 0);
  std::vector<int> input = {2, 1, 0, 3, 2, 6, 4, 8};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));

  Sdobnov_V_mergesort_Betcher_seq::MergesortBetcherSeq test(taskDataSeq);

  ASSERT_FALSE(test.validation());
}

TEST(Sdobnov_V_mergesort_Betcher_seq, InvalidOutput) {
  int size = 8;
  std::vector<int> res(size, 0);
  std::vector<int> input = {2, 1, 0, 3, 2, 6, 4, 8};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->outputs_count.emplace_back(size);

  Sdobnov_V_mergesort_Betcher_seq::MergesortBetcherSeq test(taskDataSeq);

  ASSERT_FALSE(test.validation());
}

TEST(Sdobnov_V_mergesort_Betcher_seq, SortTest8) {
  int size = 8;
  std::vector<int> res(size, 0);
  std::vector<int> input = {2, 8, 3, 9, 5, 6, 4, 0};
  std::vector<int> expected_res = {0, 2, 3, 4, 5, 6, 8, 9};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->outputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));

  Sdobnov_V_mergesort_Betcher_seq::MergesortBetcherSeq test(taskDataSeq);

  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();

  for (int i = 0; i < size; i++) {
    ASSERT_EQ(res[i], expected_res[i]);
  }
}

TEST(Sdobnov_V_mergesort_Betcher_seq, SortTestRand16) {
  int size = 16;
  std::vector<int> res(size, 0);
  std::vector<int> input = Sdobnov_V_mergesort_Betcher_seq::generate_random_vector(size, 0, 1000);
  std::vector<int> expected_res = input;
  std::sort(expected_res.begin(), expected_res.end());

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->outputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));

  Sdobnov_V_mergesort_Betcher_seq::MergesortBetcherSeq test(taskDataSeq);

  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();

  for (int i = 0; i < size; i++) {
    ASSERT_EQ(res[i], expected_res[i]);
  }
}

TEST(Sdobnov_V_mergesort_Betcher_seq, SortTestRand32) {
  int size = 32;
  std::vector<int> res(size, 0);
  std::vector<int> input = Sdobnov_V_mergesort_Betcher_seq::generate_random_vector(size, 0, 1000);
  std::vector<int> expected_res = input;
  std::sort(expected_res.begin(), expected_res.end());

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->outputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));

  Sdobnov_V_mergesort_Betcher_seq::MergesortBetcherSeq test(taskDataSeq);

  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();

  for (int i = 0; i < size; i++) {
    ASSERT_EQ(res[i], expected_res[i]);
  }
}

TEST(Sdobnov_V_mergesort_Betcher_seq, SortTestRand64) {
  int size = 64;
  std::vector<int> res(size, 0);
  std::vector<int> input = Sdobnov_V_mergesort_Betcher_seq::generate_random_vector(size, 0, 1000);
  std::vector<int> expected_res = input;
  std::sort(expected_res.begin(), expected_res.end());

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->outputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));

  Sdobnov_V_mergesort_Betcher_seq::MergesortBetcherSeq test(taskDataSeq);

  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();

  for (int i = 0; i < size; i++) {
    ASSERT_EQ(res[i], expected_res[i]);
  }
}