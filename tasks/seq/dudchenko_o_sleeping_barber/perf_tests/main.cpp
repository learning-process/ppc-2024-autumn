#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/dudchenko_o_sleeping_barber/include/ops_seq.hpp"

TEST(sleeping_barber_test, test_pipeline_run) {
  const int seats = 3;

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(seats);
  taskDataSeq->outputs_count.emplace_back(1);

  auto testTaskSequential = std::make_shared<dudchenko_o_sleeping_barber_seq::TestSleepingBarber>(taskDataSeq);

  ASSERT_TRUE(testTaskSequential->validation());

  ASSERT_TRUE(testTaskSequential->pre_processing());
  ASSERT_TRUE(testTaskSequential->run());
  ASSERT_TRUE(testTaskSequential->post_processing());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  ASSERT_TRUE(perfResults != nullptr);
}

TEST(sleeping_barber_test, test_task_run) {
  const int seats = 3;

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(seats);
  taskDataSeq->outputs_count.emplace_back(1);

  auto testTaskSequential = std::make_shared<dudchenko_o_sleeping_barber_seq::TestSleepingBarber>(taskDataSeq);

  ASSERT_TRUE(testTaskSequential->validation());

  ASSERT_TRUE(testTaskSequential->pre_processing());
  ASSERT_TRUE(testTaskSequential->run());
  ASSERT_TRUE(testTaskSequential->post_processing());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);

  ASSERT_TRUE(perfResults != nullptr);
}
