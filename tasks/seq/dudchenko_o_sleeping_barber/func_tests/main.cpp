#include <gtest/gtest.h>

#include <memory>

#include "seq/dudchenko_o_sleeping_barber/include/ops_seq.hpp"

TEST(dudchenko_o_sleeping_barber_sequential, Test_Validation1) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  dudchenko_o_sleeping_barber_seq::TestSleepingBarber testSleepingBarber(taskDataSeq);

  taskDataSeq->inputs_count = {0};
  EXPECT_FALSE(testSleepingBarber.validation());
}

TEST(dudchenko_o_sleeping_barber_sequential, Test_Validation2) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  dudchenko_o_sleeping_barber_seq::TestSleepingBarber testSleepingBarber(taskDataSeq);

  taskDataSeq->inputs_count = {1};
  EXPECT_FALSE(testSleepingBarber.validation());
}

TEST(dudchenko_o_sleeping_barber_sequential, Test_Validation3) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  dudchenko_o_sleeping_barber_seq::TestSleepingBarber testSleepingBarber(taskDataSeq);

  taskDataSeq->inputs_count = {5};
  EXPECT_TRUE(testSleepingBarber.validation());
}

TEST(dudchenko_o_sleeping_barber_sequential, Test_End_To_End1) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  const int max_waiting_chairs = 1;
  int global_res = -1;

  taskDataSeq->inputs_count.emplace_back(max_waiting_chairs);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_res));
  taskDataSeq->outputs_count.emplace_back(sizeof(global_res));

  dudchenko_o_sleeping_barber_seq::TestSleepingBarber testSleepingBarber(taskDataSeq);

  ASSERT_FALSE(testSleepingBarber.validation());
  ASSERT_TRUE(testSleepingBarber.pre_processing());
  ASSERT_TRUE(testSleepingBarber.run());
  ASSERT_TRUE(testSleepingBarber.post_processing());

  EXPECT_EQ(global_res, 0);
}

TEST(dudchenko_o_sleeping_barber_seq, Test_End_To_End2) {
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

TEST(dudchenko_o_sleeping_barber_sequential, Test_End_To_End3) {
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
