// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/zolotareva_a_count_of_words/include/ops_seq.hpp"

TEST(sequential_zolotareva_a_count_of_words_perf_test, test_pipeline_run) {
  const int count = 162;
  const std::string input =
      "I was too young to be other than awed and puzzled by Doc Marlowe when I knew him. I was only sixteen when he "
      "died. He was sixty-seven. There was that vast difference in our ages and there was a vaster difference in our "
      "backgrounds. Doc Marlowe was a medicine-show man. He had been a lot of other things, too: a circus man, the "
      "proprietor of a concession at Coney Island, a saloon-keeper; but in his fifties he had travelled around with a "
      "tent-show troupe made up of a Mexican named Chickalilli, who threw knives, and a man called Professor Jones, "
      "who played the banjo. Doc Marlowe would come out after the entertainment and harangue the crowd and sell "
      "bottles of medicine for all kinds of ailments. I found out all this about him gradually, toward the last, and "
      "after he died. When I first knew him, he represented the Wild West to me, and there was nobody I admired so "
      "much.";

  std::vector<char> in(input.begin(), input.end());
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<zolotareva_a_count_of_words_seq::TestTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(count, out[0]);
}

TEST(sequential_zolotareva_a_count_of_words_perf_test, test_task_run) {
  const int count = 162;
  const std::string input =
      "I was too young to be other than awed and puzzled by Doc Marlowe when I knew him. I was only sixteen when he "
      "died. He was sixty-seven. There was that vast difference in our ages and there was a vaster difference in our "
      "backgrounds. Doc Marlowe was a medicine-show man. He had been a lot of other things, too: a circus man, the "
      "proprietor of a concession at Coney Island, a saloon-keeper; but in his fifties he had travelled around with a "
      "tent-show troupe made up of a Mexican named Chickalilli, who threw knives, and a man called Professor Jones, "
      "who played the banjo. Doc Marlowe would come out after the entertainment and harangue the crowd and sell "
      "bottles of medicine for all kinds of ailments. I found out all this about him gradually, toward the last, and "
      "after he died. When I first knew him, he represented the Wild West to me, and there was nobody I admired so "
      "much.";

  std::vector<char> in(input.begin(), input.end());
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<zolotareva_a_count_of_words_seq::TestTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(count, out[0]);
}