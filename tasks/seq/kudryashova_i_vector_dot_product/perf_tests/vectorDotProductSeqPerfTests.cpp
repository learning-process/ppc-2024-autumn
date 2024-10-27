#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/kudryashova_i_vector_dot_product/include/vectorDotProductSeq.hpp"

TEST(kudryashova_i_vector_dot_product_seq, test_pipeline_run) {
	const int count_size = 15000000;
	
	std::vector<int> vector1 = kudryashova_i_vector_dot_product::GetRandomVector(count_size);
	std::vector<int> vector2 = kudryashova_i_vector_dot_product::GetRandomVector(count_size);

	std::vector<int> out(1, 0);

	std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
	
	taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector1.data()));
	taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector2.data()));
	
	taskDataSeq->inputs_count.emplace_back(vector1.size());
	taskDataSeq->inputs_count.emplace_back(vector2.size());
	
	taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
	taskDataSeq->outputs_count.emplace_back(out.size());
	
	auto testTaskSequential = std::make_shared<kudryashova_i_vector_dot_product::TestTaskSequential>(taskDataSeq);
	
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
	
	ASSERT_EQ(kudryashova_i_vector_dot_product::vectorDotProduct(vector1, vector2), out[0]);
}

TEST(kudryashova_i_vector_dot_product_seq, test_task_run) {
	const int count = 15000000;
	
	std::vector<int> vector1 = kudryashova_i_vector_dot_product::GetRandomVector(count);
	std::vector<int> vector2 = kudryashova_i_vector_dot_product::GetRandomVector(count);

	std::vector<int> out(1, 0);

	std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
	
	taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector1.data()));
	taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector2.data()));
	
	taskDataSeq->inputs_count.emplace_back(vector1.size());
	taskDataSeq->inputs_count.emplace_back(vector2.size());
	
	taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
	taskDataSeq->outputs_count.emplace_back(out.size());
	
	auto testTaskSequential = std::make_shared<kudryashova_i_vector_dot_product::TestTaskSequential>(taskDataSeq);

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
	
	ASSERT_EQ(kudryashova_i_vector_dot_product::vectorDotProduct(vector1, vector2), out[0]);
}