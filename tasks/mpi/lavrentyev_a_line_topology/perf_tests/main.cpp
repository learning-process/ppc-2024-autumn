#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/lavrentyev_a_line_topology/include/ops_mpi.hpp"

namespace lavrentyev_a_line_topology_mpi {

std::vector<int> generate_random_data(int count, int lower_bound = -1000, int upper_bound = 1000) {
  std::vector<int> data(count);
  std::mt19937 random_engine(std::random_device{}());
  std::uniform_int_distribution<int> distribution(lower_bound, upper_bound);
  for (int& value : data) {
    value = distribution(random_engine);
  }
  return data;
}
}  // namespace lavrentyev_a_line_topology_mpi

TEST(lavrentyev_a_line_topology_mpi, task_run) {
  boost::mpi::communicator world;

  size_t start_proc = 0;
  size_t end_proc = world.size() - 1;
  size_t num_elems = 1'000'000;

  // Настраиваем TaskData для теста
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned>(start_proc), static_cast<unsigned>(end_proc),
                             static_cast<unsigned>(num_elems)};

  std::vector<int> input_data;
  std::vector<int> output_data(num_elems, -1);
  std::vector<int> path(end_proc - start_proc + 1, -1);

  if (world.rank() == start_proc) {
    input_data = lavrentyev_a_line_topology_mpi::generate_random_data(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  }

  if (world.rank() == end_proc) {
    task_data->outputs = {reinterpret_cast<uint8_t*>(output_data.data()), reinterpret_cast<uint8_t*>(path.data())};
    task_data->outputs_count = {static_cast<unsigned>(output_data.size()), static_cast<unsigned>(path.size())};
  }

  // Создаем экземпляр задачи
  auto task = std::make_shared<lavrentyev_a_line_topology_mpi::TestMPITaskParallel>(task_data);

  // Настройка для измерения производительности
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  boost::mpi::timer perf_timer;
  perf_attr->current_timer = [&] { return perf_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);

  // Запуск теста с измерением производительности
  perf_analyzer->task_run(perf_attr, perf_results);

  if (world.rank() == end_proc) {
    // Выводим статистику производительности
    ppc::core::Perf::print_perf_statistic(perf_results);

    // Проверяем корректность результата
    ASSERT_EQ(input_data, output_data);
    for (size_t i = 0; i < path.size(); ++i) {
      ASSERT_EQ(path[i], static_cast<int>(start_proc) + static_cast<int>(i));
    }
  }
}

TEST(lavrentyev_a_line_topology_mpi, pipeline_run) {
  boost::mpi::communicator world;

  int total_elements = 10'000'000;
  int start_rank = 0;
  int end_rank = world.size() - 1;

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs_count.push_back(start_rank);
  data->inputs_count.push_back(end_rank);
  data->inputs_count.push_back(total_elements);

  std::vector<int> input_values;
  std::vector<int> result_values(total_elements);
  std::vector<int> trace_path;

  if (world.rank() == start_rank) {
    input_values = lavrentyev_a_line_topology_mpi::generate_random_data(total_elements);
    data->inputs.push_back(reinterpret_cast<uint8_t*>(input_values.data()));

    if (start_rank != end_rank) {
      world.send(end_rank, 0, input_values);
    }
  }

  if (world.rank() == end_rank) {
    if (start_rank != end_rank) {
      world.recv(start_rank, 0, input_values);
    }

    trace_path.resize(end_rank - start_rank + 1);
    data->outputs = {reinterpret_cast<uint8_t*>(result_values.data()), reinterpret_cast<uint8_t*>(trace_path.data())};
    data->outputs_count.push_back(result_values.size());
    data->outputs_count.push_back(trace_path.size());
  }

  auto parallel_task = std::make_shared<lavrentyev_a_line_topology_mpi::TestMPITaskParallel>(data);

  auto perf_attrs = std::make_shared<ppc::core::PerfAttr>();
  perf_attrs->num_running = 5;
  boost::mpi::timer task_time;
  perf_attrs->current_timer = [&] { return task_time.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(parallel_task);

  perf_analyzer->pipeline_run(perf_attrs, perf_results);

  if (world.rank() == end_rank) {
    ppc::core::Perf::print_perf_statistic(perf_results);
    ASSERT_EQ(input_values, result_values);
    for (size_t i = 0; i < trace_path.size(); ++i) {
      ASSERT_EQ(trace_path[i], start_rank + static_cast<int>(i));
    }
  }
}