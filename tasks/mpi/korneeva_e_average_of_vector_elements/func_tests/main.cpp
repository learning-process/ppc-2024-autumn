#include <gtest/gtest.h>

#include "mpi/korneeva_e_average_of_vector_elements/include/ops_mpi.hpp"

// Utility for preparing task data
template <typename iotype>
std::shared_ptr<ppc::core::TaskData> prepare_task_data(const boost::mpi::communicator& world,
                                                       const std::vector<iotype>& input, std::vector<double>& output) {
  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<iotype*>(input.data())));
    task_data->inputs_count.emplace_back(input.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    task_data->outputs_count.emplace_back(output.size());
  }
  return task_data;
}

// Function to run the test
template <typename iotype, typename TaskType>
void run_test(const std::vector<iotype>& input, boost::mpi::communicator& world, std::vector<double>& out_mpi,
              std::vector<double>& out_seq) {
  // For the MPI task
  auto task_data_mpi = prepare_task_data(world, input, out_mpi);
  TaskType task_mpi(task_data_mpi);
  ASSERT_EQ(task_mpi.validation(), true);
  task_mpi.pre_processing();
  task_mpi.run();
  task_mpi.post_processing();

  if (world.rank() == 0) {
    // For the sequential task
    auto task_data_seq = prepare_task_data(world, input, out_seq);
    korneeva_e_average_of_vector_elements_mpi::vector_average_sequential<iotype> task_seq(task_data_seq);
    ASSERT_EQ(task_seq.validation(), true);
    task_seq.pre_processing();
    task_seq.run();
    task_seq.post_processing();

    ASSERT_NEAR(out_seq[0], out_mpi[0], 1e-6);
  }
}

// Utility to generate a random vector
template <typename T>
std::vector<T> generate_random_vector(int size, T min_value, T max_value) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(min_value, max_value);

  std::vector<T> result(size);
  for (auto& el : result) {
    el = static_cast<T>(dis(gen));
  }
  return result;
}

// Tests for a monotonically increasing vector
TEST(korneeva_e_average_of_vector_elements_mpi, Average_IncreasingOrder_MPI_AllReduce) {
  const int N = 100;
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);
  std::iota(arr.begin(), arr.end(), 0);  // vector (0, 1, ..., N-1)

  run_test<int, korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<int>>(arr, world, out_mpi,
                                                                                              out_seq);
}
TEST(korneeva_e_average_of_vector_elements_mpi, Average_IncreasingOrder_my_AllReduce) {
  const int N = 100;
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);
  std::iota(arr.begin(), arr.end(), 0);  // vector (0, 1, ..., N-1)

  run_test<int, korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<int>>(arr, world, out_mpi,
                                                                                             out_seq);
}

// Tests for an empty vector
TEST(korneeva_e_average_of_vector_elements_mpi, Average_EmptyVector_MPI_AllReduce) {
  const int N = 0;
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<int, korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<int>>(arr, world, out_mpi,
                                                                                              out_seq);
}
TEST(korneeva_e_average_of_vector_elements_mpi, Average_EmptyVector_my_AllReduce) {
  const int N = 0;
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<int, korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<int>>(arr, world, out_mpi,
                                                                                             out_seq);
}

// Tests for a vector with a single element
TEST(korneeva_e_average_of_vector_elements_mpi, Average_SingleElement_MPI_AllReduce) {
  const int N = 1;
  boost::mpi::communicator world;
  std::vector<int> arr(N, 42);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<int, korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<int>>(arr, world, out_mpi,
                                                                                              out_seq);
}
TEST(korneeva_e_average_of_vector_elements_mpi, Average_SingleElement_my_AllReduce) {
  const int N = 1;
  boost::mpi::communicator world;
  std::vector<int> arr(N, 42);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<int, korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<int>>(arr, world, out_mpi,
                                                                                             out_seq);
}

// Tests for calculating the average with multiple processes
TEST(korneeva_e_average_of_vector_elements_mpi, Average_MultipleProcesses_MPI_AllReduce) {
  const int N = 4;
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);
  std::iota(arr.begin(), arr.end(), 0);

  run_test<int, korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<int>>(arr, world, out_mpi,
                                                                                              out_seq);
}
TEST(korneeva_e_average_of_vector_elements_mpi, Average_MultipleProcesses_my_AllReduce) {
  const int N = 4;
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);
  std::iota(arr.begin(), arr.end(), 0);

  run_test<int, korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<int>>(arr, world, out_mpi,
                                                                                             out_seq);
}

// Tests for a random vector of type int with 10 elements
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_10_Int_MPI_AllReduce) {
  const int N = 10;
  boost::mpi::communicator world;
  std::vector<int> arr = generate_random_vector<int>(N, -100, 100);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<int, korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<int>>(arr, world, out_mpi,
                                                                                              out_seq);
}
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_10_Int_my_AllReduce) {
  const int N = 10;
  boost::mpi::communicator world;
  std::vector<int> arr = generate_random_vector<int>(N, -100, 100);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<int, korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<int>>(arr, world, out_mpi,
                                                                                             out_seq);
}

TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_10_Float_MPI_AllReduce) {
  const int N = 10;
  boost::mpi::communicator world;
  std::vector<float> arr = generate_random_vector<float>(N, -100.0f, 100.0f);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<float, korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<float>>(arr, world, out_mpi,
                                                                                                  out_seq);
}
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_10_Float_my_AllReduce) {
  const int N = 10;
  boost::mpi::communicator world;
  std::vector<float> arr = generate_random_vector<float>(N, -100.0f, 100.0f);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<float, korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<float>>(arr, world, out_mpi,
                                                                                                 out_seq);
}

TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_10_Double_MPI_AllReduce) {
  const int N = 10;
  boost::mpi::communicator world;
  std::vector<double> arr = generate_random_vector<double>(N, -100.0, 100.0);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<double, korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<double>>(arr, world, out_mpi,
                                                                                                    out_seq);
}
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_10_Double_my_AllReduce) {
  const int N = 10;
  boost::mpi::communicator world;
  std::vector<double> arr = generate_random_vector<double>(N, -100.0, 100.0);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<double, korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<double>>(arr, world, out_mpi,
                                                                                                   out_seq);
}

// Tests for a random vector with 100 elements
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_100_Int_MPI_AllReduce) {
  const int N = 100;
  boost::mpi::communicator world;
  std::vector<int> arr = generate_random_vector<int>(N, -100, 100);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<int, korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<int>>(arr, world, out_mpi,
                                                                                              out_seq);
}
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_100_Int_my_AllReduce) {
  const int N = 100;
  boost::mpi::communicator world;
  std::vector<int> arr = generate_random_vector<int>(N, -100, 100);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<int, korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<int>>(arr, world, out_mpi,
                                                                                             out_seq);
}

TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_100_Float_MPI_AllReduce) {
  const int N = 100;
  boost::mpi::communicator world;
  std::vector<float> arr = generate_random_vector<float>(N, -100.0f, 100.0f);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<float, korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<float>>(arr, world, out_mpi,
                                                                                                  out_seq);
}
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_100_Float_my_AllReduce) {
  const int N = 100;
  boost::mpi::communicator world;
  std::vector<float> arr = generate_random_vector<float>(N, -100.0f, 100.0f);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<float, korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<float>>(arr, world, out_mpi,
                                                                                                 out_seq);
}

TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_100_Double_MPI_AllReduce) {
  const int N = 100;
  boost::mpi::communicator world;
  std::vector<double> arr = generate_random_vector<double>(N, -100.0, 100.0);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<double, korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<double>>(arr, world, out_mpi,
                                                                                                    out_seq);
}
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_100_Double_my_AllReduce) {
  const int N = 100;
  boost::mpi::communicator world;
  std::vector<double> arr = generate_random_vector<double>(N, -100.0, 100.0);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<double, korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<double>>(arr, world, out_mpi,
                                                                                                   out_seq);
}

// Tests for a random vector with 1000 elements
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_1000_Int_MPI_AllReduce) {
  const int N = 1000;
  boost::mpi::communicator world;
  std::vector<int> arr = generate_random_vector<int>(N, -1000, 1000);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<int, korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<int>>(arr, world, out_mpi,
                                                                                              out_seq);
}
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_1000_Int_my_AllReduce) {
  const int N = 1000;
  boost::mpi::communicator world;
  std::vector<int> arr = generate_random_vector<int>(N, -1000, 1000);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<int, korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<int>>(arr, world, out_mpi,
                                                                                             out_seq);
}

TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_1000_Float_MPI_AllReduce) {
  const int N = 1000;
  boost::mpi::communicator world;
  std::vector<float> arr = generate_random_vector<float>(N, -1000.0f, 1000.0f);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<float, korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<float>>(arr, world, out_mpi,
                                                                                                  out_seq);
}
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_1000_Float_my_AllReduce) {
  const int N = 1000;
  boost::mpi::communicator world;
  std::vector<float> arr = generate_random_vector<float>(N, -1000.0f, 1000.0f);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<float, korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<float>>(arr, world, out_mpi,
                                                                                                 out_seq);
}

TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_1000_Double_MPI_AllReduce) {
  const int N = 1000;
  boost::mpi::communicator world;
  std::vector<double> arr = generate_random_vector<double>(N, -1000.0, 1000.0);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<double, korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<double>>(arr, world, out_mpi,
                                                                                                    out_seq);
}
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_1000_Double_my_AllReduce) {
  const int N = 1000;
  boost::mpi::communicator world;
  std::vector<double> arr = generate_random_vector<double>(N, -1000.0, 1000.0);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<double, korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<double>>(arr, world, out_mpi,
                                                                                                   out_seq);
}

// Tests for a random vector with 10000 elements
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_10000_Int_MPI_AllReduce) {
  const int N = 10000;
  boost::mpi::communicator world;
  std::vector<int> arr = generate_random_vector<int>(N, -10000, 10000);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<int, korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<int>>(arr, world, out_mpi,
                                                                                              out_seq);
}
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_10000_Int_my_AllReduce) {
  const int N = 10000;
  boost::mpi::communicator world;
  std::vector<int> arr = generate_random_vector<int>(N, -10000, 10000);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<int, korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<int>>(arr, world, out_mpi,
                                                                                             out_seq);
}

TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_10000_Float_MPI_AllReduce) {
  const int N = 10000;
  boost::mpi::communicator world;
  std::vector<float> arr = generate_random_vector<float>(N, -10000.0f, 10000.0f);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<float, korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<float>>(arr, world, out_mpi,
                                                                                                  out_seq);
}
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_10000_Float_my_AllReduce) {
  const int N = 10000;
  boost::mpi::communicator world;
  std::vector<float> arr = generate_random_vector<float>(N, -10000.0f, 10000.0f);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<float, korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<float>>(arr, world, out_mpi,
                                                                                                 out_seq);
}

TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_10000_Double_MPI_AllReduce) {
  const int N = 10000;
  boost::mpi::communicator world;
  std::vector<double> arr = generate_random_vector<double>(N, -10000.0, 10000.0);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<double, korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<double>>(arr, world, out_mpi,
                                                                                                    out_seq);
}
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_10000_Double_my_AllReduce) {
  const int N = 10000;
  boost::mpi::communicator world;
  std::vector<double> arr = generate_random_vector<double>(N, -10000.0, 10000.0);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<double, korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<double>>(arr, world, out_mpi,
                                                                                                   out_seq);
}

// Tests for a random vector with 100000 elements
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_100000_Int_MPI_AllReduce) {
  const int N = 100000;
  boost::mpi::communicator world;
  std::vector<int> arr = generate_random_vector<int>(N, -100000, 100000);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<int, korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<int>>(arr, world, out_mpi,
                                                                                              out_seq);
}
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_100000_Int_my_AllReduce) {
  const int N = 100000;
  boost::mpi::communicator world;
  std::vector<int> arr = generate_random_vector<int>(N, -100000, 100000);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<int, korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<int>>(arr, world, out_mpi,
                                                                                             out_seq);
}

TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_100000_Float_MPI_AllReduce) {
  const int N = 100000;
  boost::mpi::communicator world;
  std::vector<float> arr = generate_random_vector<float>(N, -100000.0f, 100000.0f);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<float, korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<float>>(arr, world, out_mpi,
                                                                                                  out_seq);
}
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_100000_Float_my_AllReduce) {
  const int N = 100000;
  boost::mpi::communicator world;
  std::vector<float> arr = generate_random_vector<float>(N, -100000.0f, 100000.0f);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<float, korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<float>>(arr, world, out_mpi,
                                                                                                 out_seq);
}

TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_100000_Double_MPI_AllReduce) {
  const int N = 100000;
  boost::mpi::communicator world;
  std::vector<double> arr = generate_random_vector<double>(N, -100000.0, 100000.0);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<double, korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<double>>(arr, world, out_mpi,
                                                                                                    out_seq);
}
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_100000_Double_my_AllReduce) {
  const int N = 100000;
  boost::mpi::communicator world;
  std::vector<double> arr = generate_random_vector<double>(N, -100000.0, 100000.0);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<double, korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<double>>(arr, world, out_mpi,
                                                                                                   out_seq);
}

// Tests for a random vector with 1000000 elements
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_1000000_Int_MPI_AllReduce) {
  const int N = 1000000;
  boost::mpi::communicator world;
  std::vector<int> arr = generate_random_vector<int>(N, -100000, 100000);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<int, korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<int>>(arr, world, out_mpi,
                                                                                              out_seq);
}
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_1000000_Int_my_AllReduce) {
  const int N = 1000000;
  boost::mpi::communicator world;
  std::vector<int> arr = generate_random_vector<int>(N, -100000, 100000);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<int, korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<int>>(arr, world, out_mpi,
                                                                                             out_seq);
}

TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_1000000_Float_MPI_AllReduce) {
  const int N = 1000000;
  boost::mpi::communicator world;
  std::vector<float> arr = generate_random_vector<float>(N, -100000.0f, 100000.0f);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<float, korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<float>>(arr, world, out_mpi,
                                                                                                  out_seq);
}
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_1000000_Float_my_AllReduce) {
  const int N = 1000000;
  boost::mpi::communicator world;
  std::vector<float> arr = generate_random_vector<float>(N, -100000.0f, 100000.0f);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<float, korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<float>>(arr, world, out_mpi,
                                                                                                 out_seq);
}

TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_1000000_Double_MPI_AllReduce) {
  const int N = 1000000;
  boost::mpi::communicator world;
  std::vector<double> arr = generate_random_vector<double>(N, -100000.0, 100000.0);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<double, korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<double>>(arr, world, out_mpi,
                                                                                                    out_seq);
}
TEST(korneeva_e_average_of_vector_elements_mpi, RandomVector_1000000_Double_my_AllReduce) {
  const int N = 1000000;
  boost::mpi::communicator world;
  std::vector<double> arr = generate_random_vector<double>(N, -100000.0, 100000.0);
  std::vector<double> out_mpi(1);
  std::vector<double> out_seq(1);

  run_test<double, korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<double>>(arr, world, out_mpi,
                                                                                                   out_seq);
}
