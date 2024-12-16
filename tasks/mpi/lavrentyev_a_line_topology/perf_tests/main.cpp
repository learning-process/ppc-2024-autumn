#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/lavrentyev_a_line_topology/include/ops_mpi.hpp"

std::vector<int> lavrentyrev_generate_random_vector(size_t size) {
  std::srand(static_cast<unsigned>(std::time(nullptr)));
  std::vector<int> vec(size);
  for (size_t i = 0; i < size; i++) {
    vec[i] = std::rand() % 1000;
  }
  return vec;
}

TEST(lavrentyev_a_line_topology_mpi, pipeline_run) {
  boost::mpi::communicator world;

  size_t start_proc = 0;
  size_t end_proc = world.size() > 1 ? static_cast<size_t>(world.size() - 1) : 0;
  size_t num_elems = 10000;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned int>(start_proc), static_cast<unsigned int>(end_proc),
                             static_cast<unsigned int>(num_elems)};

  std::vector<int> input_data;
  std::vector<int> output_data(num_elems, -1);
  std::vector<int> received_path(end_proc - start_proc + 1, -1);

  if (static_cast<size_t>(world.rank()) == start_proc) {
    input_data = lavrentyrev_generate_random_vector(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  }

  if (static_cast<size_t>(world.rank()) == end_proc) {
    task_data->outputs = {reinterpret_cast<uint8_t*>(output_data.data()),
                          reinterpret_cast<uint8_t*>(received_path.data())};
    task_data->outputs_count = {static_cast<unsigned int>(output_data.size()),
                                static_cast<unsigned int>(received_path.size())};
  }

  auto testMpiTaskParallel = std::make_shared<lavrentyev_a_line_topology_mpi::TestMPITaskParallel>(task_data);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (static_cast<size_t>(world.rank()) == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(lavrentyev_a_line_topology_mpi, task_run) {
  boost::mpi::communicator world;

  size_t start_proc = 0;
  size_t end_proc = world.size() > 1 ? static_cast<size_t>(world.size() - 1) : 0;
  size_t num_elems = 10000;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned int>(start_proc), static_cast<unsigned int>(end_proc),
                             static_cast<unsigned int>(num_elems)};

  std::vector<int> input_data;
  std::vector<int> output_data(num_elems, -1);
  std::vector<int> received_path(end_proc - start_proc + 1, -1);

  if (static_cast<size_t>(world.rank()) == start_proc) {
    input_data = lavrentyrev_generate_random_vector(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  }

  if (static_cast<size_t>(world.rank()) == end_proc) {
    task_data->outputs = {reinterpret_cast<uint8_t*>(output_data.data()),
                          reinterpret_cast<uint8_t*>(received_path.data())};
    task_data->outputs_count = {static_cast<unsigned int>(output_data.size()),
                                static_cast<unsigned int>(received_path.size())};
  }

  auto testMpiTaskParallel = std::make_shared<lavrentyev_a_line_topology_mpi::TestMPITaskParallel>(task_data);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (static_cast<size_t>(world.rank()) == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}