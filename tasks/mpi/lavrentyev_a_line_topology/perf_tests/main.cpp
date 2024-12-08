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

  int num_elems = 1000000;
  int start_proc = 0;
  int end_proc = world.size() - 1;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(start_proc);
  task_data->inputs_count.push_back(end_proc);
  task_data->inputs_count.push_back(num_elems);

  std::vector<int> input_data;
  std::vector<int> output_data(num_elems);
  std::vector<int> processing_sequence;

  MPI_Request req_send;
  MPI_Request req_recv;

  if (world.rank() == start_proc) {
    input_data = lavrentyev_a_line_topology_mpi::generate_random_data(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));

    if (start_proc != end_proc) {
      MPI_Isend(input_data.data(), input_data.size(), MPI_INT, end_proc, 0, MPI_COMM_WORLD, &req_send);
      MPI_Wait(&req_send, MPI_STATUS_IGNORE);
    }
  }

  if (world.rank() == end_proc) {
    if (start_proc != end_proc) {
      input_data.resize(num_elems);
      MPI_Irecv(input_data.data(), input_data.size(), MPI_INT, start_proc, 0, MPI_COMM_WORLD, &req_recv);
      MPI_Wait(&req_recv, MPI_STATUS_IGNORE);
    }

    processing_sequence.resize(end_proc - start_proc + 1);
    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    task_data->outputs_count.push_back(output_data.size());
    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(processing_sequence.data()));
    task_data->outputs_count.push_back(processing_sequence.size());
  }

  auto mpi_task = std::make_shared<lavrentyev_a_line_topology_mpi::TestMPITaskParallel>(task_data);

  auto performance_attr = std::make_shared<ppc::core::PerfAttr>();
  performance_attr->num_running = 10;
  boost::mpi::timer performance_timer;
  performance_attr->current_timer = [&] { return performance_timer.elapsed(); };

  auto performance_results = std::make_shared<ppc::core::PerfResults>();
  auto performance_analyzer = std::make_shared<ppc::core::Perf>(mpi_task);

  performance_analyzer->task_run(performance_attr, performance_results);

  if (world.rank() == end_proc) {
    ppc::core::Perf::print_perf_statistic(performance_results);

    for (int i = 0; i < num_elems; i++) {
      ASSERT_EQ(output_data[i], input_data[i]);
    }
    for (size_t i = 0; i < processing_sequence.size(); ++i) {
      ASSERT_EQ(processing_sequence[i], start_proc + static_cast<int>(i));
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

  MPI_Request req_send;
  MPI_Request req_recv;

  if (world.rank() == start_rank) {
    input_values = lavrentyev_a_line_topology_mpi::generate_random_data(total_elements);
    data->inputs.push_back(reinterpret_cast<uint8_t*>(input_values.data()));

    if (start_rank != end_rank) {
      MPI_Isend(input_values.data(), input_values.size(), MPI_INT, end_rank, 0, MPI_COMM_WORLD, &req_send);
      MPI_Wait(&req_send, MPI_STATUS_IGNORE);
    }
  }

  if (world.rank() == end_rank) {
    if (start_rank != end_rank) {
      input_values.resize(total_elements);
      MPI_Irecv(input_values.data(), input_values.size(), MPI_INT, start_rank, 0, MPI_COMM_WORLD, &req_recv);
      MPI_Wait(&req_recv, MPI_STATUS_IGNORE);
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
