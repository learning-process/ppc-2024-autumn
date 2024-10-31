#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/petrov_o_num_of_alternations_signs/include/ops_mpi.hpp" // Обновленный include path
#include <boost/mpi.hpp>


// Вспомогательная функция для создания и запуска задачи
template <typename TaskType>
void runPerformanceTest(int size, int num_running) {
    std::vector<int> in(size);
    std::iota(in.begin(), in.end(), 1);
    for (size_t i = 0; i < in.size(); ++i) {
        if (i % 2 != 0) {
            in[i] *= -1;
        }
    }
    std::vector<int> out(1);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskData->inputs_count.emplace_back(in.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskData->outputs_count.emplace_back(out.size());

    // Create Task
    auto task = std::make_shared<TaskType>(taskData);

    // Create Perf attributes
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = num_running;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perfAttr->current_timer = [&] {
        auto current_time_point = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
        return static_cast<double>(duration) * 1e-9;
    };

    // Create and init perf results
    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    // Create Perf analyzer
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);


    // Запуск теста производительности (pipeline_run или task_run)
}


TEST(petrov_o_num_of_alternations_signs_seq, test_pipeline_run) {
    boost::mpi::communicator world;
    runPerformanceTest<petrov_o_num_of_alternations_signs_mpi::SequentialTask>(100000, 10);
}


TEST(petrov_o_num_of_alternations_signs_seq, test_task_run) {
    boost::mpi::communicator world;
    runPerformanceTest<petrov_o_num_of_alternations_signs_mpi::SequentialTask>(100000, 10);
}


TEST(petrov_o_num_of_alternations_signs_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  runPerformanceTest<petrov_o_num_of_alternations_signs_mpi::ParallelTask>(100000, 10);

}

TEST(petrov_o_num_of_alternations_signs_mpi, test_task_run) {
  boost::mpi::communicator world;
  runPerformanceTest<petrov_o_num_of_alternations_signs_mpi::ParallelTask>(100000, 10);
}