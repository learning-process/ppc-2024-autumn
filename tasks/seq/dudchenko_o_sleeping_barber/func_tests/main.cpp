#include <gtest/gtest.h>

#include <memory>

#include "seq/dudchenko_o_sleeping_barber/include/ops_seq.hpp"

TEST(dudchenko_o_sleeping_barber_sequential, validation_test_1) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  dudchenko_o_sleeping_barber_seq::TestSleepingBarber testSleepingBarber(taskDataSeq);

  taskDataSeq->inputs_count = {0};
  EXPECT_FALSE(testSleepingBarber.validation());
}

TEST(dudchenko_o_sleeping_barber_sequential, validation_test_2) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  dudchenko_o_sleeping_barber_seq::TestSleepingBarber testSleepingBarber(taskDataSeq);

  taskDataSeq->inputs_count = {1};
  EXPECT_FALSE(testSleepingBarber.validation());
}

TEST(dudchenko_o_sleeping_barber_sequential, validation_test_3) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  dudchenko_o_sleeping_barber_seq::TestSleepingBarber testSleepingBarber(taskDataSeq);

  taskDataSeq->inputs_count = {5};
  EXPECT_TRUE(testSleepingBarber.validation());
}

TEST(dudchenko_o_sleeping_barber_seq, functional_test_1) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  const int max_waiting_chairs = 3;
  int global_res = -1;

  taskDataSeq->inputs_count.emplace_back(max_waiting_chairs);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_res));
  taskDataSeq->outputs_count.emplace_back(sizeof(global_res));

  dudchenko_o_sleeping_barber_seq::TestSleepingBarber testSleepingBarber(taskDataSeq);

  ASSERT_TRUE(testSleepingBarber.validation());
  ASSERT_TRUE(testSleepingBarber.pre_processing());
  ASSERT_TRUE(testSleepingBarber.run());
  ASSERT_TRUE(testSleepingBarber.post_processing());

  EXPECT_EQ(global_res, 0);
}

TEST(dudchenko_o_sleeping_barber_sequential, functional_test_2) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  const int max_waiting_chairs = 1024;
  int global_res = -1;

  taskDataSeq->inputs_count.emplace_back(max_waiting_chairs);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_res));
  taskDataSeq->outputs_count.emplace_back(sizeof(global_res));

  dudchenko_o_sleeping_barber_seq::TestSleepingBarber testSleepingBarber(taskDataSeq);

  ASSERT_TRUE(testSleepingBarber.validation());
  ASSERT_TRUE(testSleepingBarber.pre_processing());
  ASSERT_TRUE(testSleepingBarber.run());
  ASSERT_TRUE(testSleepingBarber.post_processing());

  EXPECT_EQ(global_res, 0);
}
