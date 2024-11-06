
#include <gtest/gtest.h>

#include <vector>
#include <random>

#include "core/perf/include/perf.hpp"
#include "seq/komshina_d_min_of_vector_elements/include/ops_seq.hpp"




TEST(komshina_d_min_of_vector_elements_seq, test_pipeline_run) {
  const int count = 5'000'000; 
  const int start = 500; 
  const int min = -10;

  std::vector<int> in(count, start);

  
  std::random_device dev;
  std::mt19937 gen(dev());

 
  for (int i = 0; i < count - 1; i++) {
    in[i] = gen() % 1000;  
  }
  in[count - 10] = min;  

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential>(taskDataSeq);

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

  

  ASSERT_EQ(min, out[0]);
}

TEST(komshina_d_min_of_vector_elements_seq, test_task_run) {

 const int count = 5'000'000;
  const int start = 500;
  const int min = -10;

  std::vector<int> in(count, start);

  std::random_device dev;
  std::mt19937 gen(dev());

  for (int i = 0; i < count - 1; i++) {
    in[i] = gen() % 1000;
  }

  in[count - 10] = min;

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential>(taskDataSeq);

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

  ASSERT_EQ(min, out[0]);
}

