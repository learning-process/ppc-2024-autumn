#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/tarakanov_d_ring_topology/include/ops_mpi.hpp"

namespace tarakanov_d_ring_topology_mpi_test {

std::vector<int> generate_random_vector(int size) {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::vector<int> result(size);
  for (int i = 0; i < size; i++) {
    result[i] = static_cast<int>(generator() % 200 - 100);
  }
  return result;
}

}  // namespace tarakanov_d_ring_topology_mpi_test

namespace {

void initialize_task_data(boost::mpi::communicator& world, std::shared_ptr<ppc::core::TaskData>& task_data,
                          std::vector<int>& initial_data, std::vector<int>& final_data, int vector_length) {
  if (world.rank() == 0) {
    initial_data = tarakanov_d_ring_topology_mpi_test::generate_random_vector(vector_length);
    final_data.resize(vector_length);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(initial_data.data()));
    task_data->inputs_count.emplace_back(initial_data.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(final_data.data()));
    task_data->outputs_count.emplace_back(final_data.size());
  }
}

void execute_task_and_validate(boost::mpi::communicator& world, const std::vector<int>& initial_data,
                               const std::vector<int>& final_data,
                               const std::shared_ptr<ppc::core::TaskData>& task_data) {
  tarakanov_d_test_task_mpi::TestMPITaskParallel task_parallel(task_data);
  ASSERT_TRUE(task_parallel.validation());
  task_parallel.pre_processing();
  task_parallel.run();
  task_parallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(initial_data, final_data);
  }
}

void run_test_case(int vector_length) {
  boost::mpi::communicator world;
  std::vector<int> initial_data;
  std::vector<int> final_data;
  auto task_data = std::make_shared<ppc::core::TaskData>();

  initialize_task_data(world, task_data, initial_data, final_data, vector_length);
  execute_task_and_validate(world, initial_data, final_data, task_data);
}

}  // namespace

TEST(tarakanov_d_ring_topology_mpi_test, test_size_0) { run_test_case(0); }

TEST(tarakanov_d_ring_topology_mpi_test, test_size_1) { run_test_case(1); }

TEST(tarakanov_d_ring_topology_mpi_test, test_size_50) { run_test_case(50); }

TEST(tarakanov_d_ring_topology_mpi_test, test_size_150) { run_test_case(150); }

TEST(tarakanov_d_ring_topology_mpi_test, test_size_300) { run_test_case(300); }

TEST(tarakanov_d_ring_topology_mpi_test, test_size_10) { run_test_case(10); }

TEST(tarakanov_d_ring_topology_mpi_test, test_size_255) { run_test_case(255); }

TEST(tarakanov_d_ring_topology_mpi_test, test_size_512) { run_test_case(512); }

TEST(tarakanov_d_ring_topology_mpi_test, test_size_64) { run_test_case(64); }

TEST(tarakanov_d_ring_topology_mpi_test, test_size_1023) { run_test_case(1023); }
