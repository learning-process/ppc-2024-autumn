#include <gtest/gtest.h>

#include <core/perf/include/perf.hpp>
#include <seq/guseynov_e_check_lex_order_of_two_string/include/ops_seq.hpp>
#include <vector>

TEST(guseynov_e_check_lex_order_of_two_string_seq, test_pipeline_run) {
  // create data
  std::vector<std::vector<char>> in(2, std::vector<char>(20000000, 'a'));
  in[1].push_back('a');
  std::vector<int> out(1, 0);

  // create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[1].data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(in[0].size());
  taskDataSeq->inputs_count.emplace_back(in[1].size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // create Task
  auto testTaskSequantial =
      std::make_shared<guseynov_e_check_lex_order_of_two_string_seq::TestTaskSequential>(taskDataSeq);

  // Create perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequantial);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(out[0], 1);
}

TEST(guseynov_e_check_lex_order_of_two_string_seq, test_task_run) {
  // create data
  std::vector<std::vector<char>> in(2, std::vector<char>(20000000, 'a'));
  in[1].push_back('a');
  std::vector<int> out(1, 0);

  // create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[1].data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(in[0].size());
  taskDataSeq->inputs_count.emplace_back(in[1].size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // create Task
  auto testTaskSequantial =
      std::make_shared<guseynov_e_check_lex_order_of_two_string_seq::TestTaskSequential>(taskDataSeq);

  // Create perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequantial);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(out[0], 1);
}
