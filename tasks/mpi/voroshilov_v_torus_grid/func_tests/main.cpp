#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/voroshilov_v_torus_grid/include/ops_mpi.hpp"

int generate_rank(int world_size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  return gen() % world_size;
}

std::vector<char> generate_data(size_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<char> vector(size);

  std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#$%&(){}[]*+-/";

  for (size_t i = 0; i < vector.size(); i++) {
    int number = gen() % charset.length();
    vector[i] = charset[number];
  }

  return vector;
}

std::vector<int> calculate_expected_path(int source_id, int destination_id, int world_size) {
  std::vector<int> path;

  int grid = sqrt(world_size);

  if ((source_id >= world_size) || (destination_id >= world_size)) {
    path.push_back(-1);
    return path;
  }

  int next_id = source_id;
  path.push_back(next_id);
  while (next_id != destination_id) {
    next_id = voroshilov_v_torus_grid_mpi::select_path_proc(next_id, destination_id, grid);
    path.push_back(next_id);
  }

  return path;
}

bool check_grid(int world_size) {
  bool res = true;
  int n = sqrt(world_size);
  if (n * n != world_size) {
    res = false;
  }
  return res;
}

bool validation_test(int data_count, int source_id, int destination_id) {
  boost::mpi::communicator world;

  int data_size = data_count;
  int src_proc = source_id;
  int dst_proc = destination_id;

  // This task requires a "square" number of processes
  if (!check_grid(world.size())) {
    return false;
  }

  std::vector<char> input_data;
  std::vector<char> output_data;
  std::vector<int> expected_path;
  std::vector<int> output_path;

  if (world.rank() == src_proc) {
    input_data = std::vector<char>(data_size);
  }
  if (world.rank() == dst_proc) {
    input_data = std::vector<char>(data_size);
    output_data = std::vector<char>(data_size);
    expected_path = calculate_expected_path(src_proc, dst_proc, world.size());
    output_path = std::vector<int>(expected_path.size());
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(src_proc);
  taskDataPar->inputs_count.emplace_back(dst_proc);

  if (world.rank() == src_proc) {
    taskDataPar->inputs_count.emplace_back(input_data.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  }

  if (world.rank() == dst_proc) {
    taskDataPar->inputs_count.emplace_back(input_data.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));

    taskDataPar->outputs_count.emplace_back(output_data.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_path.data()));
  }

  voroshilov_v_torus_grid_mpi::TorusGridTaskParallel torusGridTaskParallel(taskDataPar);

  return torusGridTaskParallel.validation();
}

bool full_test(int data_count, int source_id, int destination_id) {
  boost::mpi::communicator world;

  int data_size = data_count;
  int src_proc = source_id;
  int dst_proc = destination_id;

  // This task requires a "square" number of processes
  if (!check_grid(world.size())) {
    return true;
  }

  struct Func_tags {
    int send_generated_data = 10;
    int send_flag_data = 11;
    int send_flag_path = 12;
  } func_tags;

  std::vector<char> input_data;
  std::vector<char> output_data;
  std::vector<int> expected_path;
  std::vector<int> output_path;

  if (world.rank() == src_proc) {
    input_data = generate_data(data_size);
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.send(dst_proc, func_tags.send_generated_data, input_data.data(), input_data.size());
    }
  }
  if (world.rank() == dst_proc) {
    input_data = std::vector<char>(data_size);
    output_data = std::vector<char>(data_size);
    expected_path = calculate_expected_path(src_proc, dst_proc, world.size());
    output_path = std::vector<int>(expected_path.size());
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.recv(src_proc, func_tags.send_generated_data, input_data.data(), input_data.size());
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(src_proc);
  taskDataPar->inputs_count.emplace_back(dst_proc);

  if (world.rank() == src_proc) {
    taskDataPar->inputs_count.emplace_back(input_data.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  }

  if (world.rank() == dst_proc) {
    taskDataPar->inputs_count.emplace_back(input_data.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));

    taskDataPar->outputs_count.emplace_back(output_data.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_path.data()));
  }

  voroshilov_v_torus_grid_mpi::TorusGridTaskParallel torusGridTaskParallel(taskDataPar);
  if (!torusGridTaskParallel.validation()) {
    return false;
  }
  torusGridTaskParallel.pre_processing();
  torusGridTaskParallel.run();
  torusGridTaskParallel.post_processing();

  // ASSERT_EQ() doesnt work if world.rank() != 0 so decided send to process 0 and check there
  if (world.rank() == dst_proc) {
    bool flag_data = true;
    for (size_t i = 0; i < output_data.size(); i++) {
      if (output_data[i] != input_data[i]) {
        flag_data = false;
      }
    }
    bool flag_path = true;
    for (size_t i = 0; i < output_path.size(); i++) {
      if (output_path[i] != expected_path[i]) {
        flag_path = false;
      }
    }
    if ((world.size() > 1) && (dst_proc != 0)) {
      world.send(0, func_tags.send_flag_data, flag_data);
      world.send(0, func_tags.send_flag_path, flag_path);
    }
  }

  if (world.rank() == 0) {
    bool flg_data = true;
    bool flg_path = true;
    if ((world.size() > 1) && (dst_proc != 0)) {
      world.recv(dst_proc, func_tags.send_flag_data, flg_data);
      world.recv(dst_proc, func_tags.send_flag_path, flg_path);
    }
    if (!flg_data) {
      return false;
    }
    if (!flg_path) {
      return false;
    }
  }

  return true;
}

TEST(voroshilov_v_torus_grid_mpi_func, test_validation_src_process_not_exists_mpi) {
  boost::mpi::communicator world_test;

  int data_sz = 100000;
  int src_proc = world_test.size();
  int dst_proc = 0;

  ASSERT_EQ(validation_test(data_sz, src_proc, dst_proc), false);
}

TEST(voroshilov_v_torus_grid_mpi_func, test_validation_dst_process_not_exists_mpi) {
  boost::mpi::communicator world_test;

  int data_sz = 100000;
  int src_proc = 0;
  int dst_proc = world_test.size();

  ASSERT_EQ(validation_test(data_sz, src_proc, dst_proc), false);
}

TEST(voroshilov_v_torus_grid_mpi_func, test_run_first_to_first_mpi) {
  boost::mpi::communicator world_test;

  int data_sz = 100001;
  int src_proc = 0;
  int dst_proc = 0;

  ASSERT_EQ(full_test(data_sz, src_proc, dst_proc), true);
}

TEST(voroshilov_v_torus_grid_mpi_func, test_run_first_to_middle_mpi) {
  boost::mpi::communicator world_test;

  int data_sz = 65536;  // 2 ^ 16
  int src_proc = 0;
  int dst_proc = world_test.size() / 2;

  ASSERT_EQ(full_test(data_sz, src_proc, dst_proc), true);
}

TEST(voroshilov_v_torus_grid_mpi_func, test_run_first_to_last_mpi) {
  boost::mpi::communicator world_test;

  int data_sz = 100000;
  int src_proc = 0;
  int dst_proc = world_test.size() - 1;

  ASSERT_EQ(full_test(data_sz, src_proc, dst_proc), true);
}

TEST(voroshilov_v_torus_grid_mpi_func, test_run_middle_to_first_mpi) {
  boost::mpi::communicator world_test;

  int data_sz = 99989;  // prime number
  int src_proc = world_test.size() / 2;
  int dst_proc = 0;

  ASSERT_EQ(full_test(data_sz, src_proc, dst_proc), true);
}

TEST(voroshilov_v_torus_grid_mpi_func, test_run_middle_to_middle_mpi) {
  boost::mpi::communicator world_test;

  int data_sz = 100001;
  int src_proc = world_test.size() / 2;
  int dst_proc = world_test.size() / 2;

  ASSERT_EQ(full_test(data_sz, src_proc, dst_proc), true);
}

TEST(voroshilov_v_torus_grid_mpi_func, test_run_middle_to_last_mpi) {
  boost::mpi::communicator world_test;

  int data_sz = 65536;  // 2 ^ 16
  int src_proc = world_test.size() / 2;
  int dst_proc = world_test.size() - 1;

  ASSERT_EQ(full_test(data_sz, src_proc, dst_proc), true);
}

TEST(voroshilov_v_torus_grid_mpi_func, test_run_last_to_first_mpi) {
  boost::mpi::communicator world_test;

  int data_sz = 100000;
  int src_proc = world_test.size() - 1;
  int dst_proc = 0;

  ASSERT_EQ(full_test(data_sz, src_proc, dst_proc), true);
}

TEST(voroshilov_v_torus_grid_mpi_func, test_run_last_to_middle_mpi) {
  boost::mpi::communicator world_test;

  int data_sz = 99989;  // prime number
  int src_proc = world_test.size() - 1;
  int dst_proc = world_test.size() / 2;

  ASSERT_EQ(full_test(data_sz, src_proc, dst_proc), true);
}

TEST(voroshilov_v_torus_grid_mpi_func, test_run_last_to_last_mpi) {
  boost::mpi::communicator world_test;

  int data_sz = 100001;
  int src_proc = world_test.size() - 1;
  int dst_proc = world_test.size() - 1;

  ASSERT_EQ(full_test(data_sz, src_proc, dst_proc), true);
}

TEST(voroshilov_v_torus_grid_mpi_func, test_run_random_to_random_2powerN_mpi) {
  boost::mpi::communicator world_test;

  int data_sz = 65536;  // 2 ^ 16
  int src_proc = 0;
  int dst_proc = 0;

  if (world_test.rank() == 0) {
    src_proc = generate_rank(world_test.size());
    dst_proc = generate_rank(world_test.size());
  }
  // broadcast() because in other way each process generates its own local src_proc and dst_proc
  boost::mpi::broadcast(world_test, src_proc, 0);
  boost::mpi::broadcast(world_test, dst_proc, 0);

  ASSERT_EQ(full_test(data_sz, src_proc, dst_proc), true);
}

TEST(voroshilov_v_torus_grid_mpi_func, test_run_random_to_random_prime_mpi) {
  boost::mpi::communicator world_test;

  int data_sz = 99989;  // prime number
  int src_proc = 0;
  int dst_proc = 0;

  if (world_test.rank() == 0) {
    src_proc = generate_rank(world_test.size());
    dst_proc = generate_rank(world_test.size());
  }
  // broadcast() because in other way each process generates its own local src_proc and dst_proc
  boost::mpi::broadcast(world_test, src_proc, 0);
  boost::mpi::broadcast(world_test, dst_proc, 0);

  ASSERT_EQ(full_test(data_sz, src_proc, dst_proc), true);
}
