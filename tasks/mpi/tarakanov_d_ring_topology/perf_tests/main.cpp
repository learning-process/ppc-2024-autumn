#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/tarakanov_d_ring_topology/include/ops_mpi.hpp"

namespace tarakanov_d_ring_topology_mpi_test {

std::vector<int> create_random_vector(int sz) {
  std::random_device device;
  std::mt19937 engine(device());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = static_cast<int>(engine() % 200 - 100);
  }
  return vec;
}

}  // namespace tarakanov_d_ring_topology_mpi_test

namespace {

void initialize_data(boost::mpi::communicator& world, std::vector<int>& initial_data, std::vector<int>& final_data,
                     std::shared_ptr<ppc::core::TaskData>& parallel_task_data) {
  if (world.rank() == 0) {
    const int vec_size = 2048;
    initial_data = tarakanov_d_ring_topology_mpi_test::create_random_vector(vec_size);
    final_data.resize(vec_size);
    parallel_task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(initial_data.data()));
    parallel_task_data->inputs_count.emplace_back(initial_data.size());
    parallel_task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(final_data.data()));
    parallel_task_data->outputs_count.emplace_back(final_data.size());
  }
}

void setup_performance_analysis(const std::shared_ptr<tarakanov_d_test_task_mpi::TestMPITaskParallel>& mpi_task,
                                std::shared_ptr<ppc::core::PerfAttr>& perf_attributes,
                                std::shared_ptr<ppc::core::PerfResults>& perf_res, boost::mpi::timer& elapsed_timer) {
  perf_attributes = std::make_shared<ppc::core::PerfAttr>();
  perf_attributes->num_running = 5;
  perf_attributes->current_timer = [&] { return elapsed_timer.elapsed(); };

  perf_res = std::make_shared<ppc::core::PerfResults>();
}

void validate_and_compare_results(const boost::mpi::communicator& world, const std::vector<int>& initial_data,
                                  const std::vector<int>& final_data,
                                  const std::shared_ptr<ppc::core::PerfResults>& perf_res) {
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perf_res);
    ASSERT_EQ(initial_data, final_data);
  }
}

}  // namespace

TEST(tarakanov_d_ring_topology_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> initial_data;
  std::vector<int> final_data;
  auto parallel_task_data = std::make_shared<ppc::core::TaskData>();

  initialize_data(world, initial_data, final_data, parallel_task_data);

  auto mpi_task = std::make_shared<tarakanov_d_test_task_mpi::TestMPITaskParallel>(parallel_task_data);
  ASSERT_TRUE(mpi_task->validation());
  mpi_task->pre_processing();
  mpi_task->run();
  mpi_task->post_processing();

  std::shared_ptr<ppc::core::PerfAttr> perf_attributes;
  std::shared_ptr<ppc::core::PerfResults> perf_res;
  boost::mpi::timer elapsed_timer;

  setup_performance_analysis(mpi_task, perf_attributes, perf_res, elapsed_timer);

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(mpi_task);
  perf_analyzer->pipeline_run(perf_attributes, perf_res);

  validate_and_compare_results(world, initial_data, final_data, perf_res);
}

TEST(tarakanov_d_ring_topology_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> initial_data;
  std::vector<int> final_data;
  auto parallel_task_data = std::make_shared<ppc::core::TaskData>();

  const int small_vec_size = 128;
  if (world.rank() == 0) {
    initial_data = tarakanov_d_ring_topology_mpi_test::create_random_vector(small_vec_size);
    final_data.resize(small_vec_size);
    parallel_task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(initial_data.data()));
    parallel_task_data->inputs_count.emplace_back(initial_data.size());
    parallel_task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(final_data.data()));
    parallel_task_data->outputs_count.emplace_back(final_data.size());
  }

  auto mpi_task = std::make_shared<tarakanov_d_test_task_mpi::TestMPITaskParallel>(parallel_task_data);
  ASSERT_TRUE(mpi_task->validation());
  mpi_task->pre_processing();
  mpi_task->run();
  mpi_task->post_processing();

  std::shared_ptr<ppc::core::PerfAttr> perf_attributes;
  std::shared_ptr<ppc::core::PerfResults> perf_res;
  boost::mpi::timer elapsed_timer;

  setup_performance_analysis(mpi_task, perf_attributes, perf_res, elapsed_timer);

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(mpi_task);
  perf_analyzer->task_run(perf_attributes, perf_res);

  validate_and_compare_results(world, initial_data, final_data, perf_res);
}
