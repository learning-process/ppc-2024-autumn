#include <gtest/gtest.h>

#include "mpi/korneeva_e_num_of_orderly_violations/include/ops_mpi.hpp"

// Test for a single-element vector, expecting no violations
TEST(korneeva_e_num_of_orderly_violations_mpi, NoViolations_SingleElement) {
  const int N = 1;  // Size of the vector
  boost::mpi::communicator world;
  int rank = world.rank();
  std::vector<int> arr(N, 42);  // Initialize vector with a single value
  std::vector<int> out(1);
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    data_seq->inputs_count.emplace_back(arr.size());
    data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    data_seq->outputs_count.emplace_back(1);
  }

  korneeva_e_num_of_orderly_violations_mpi::num_of_orderly_violations<int, int> task(data_seq);
  ASSERT_TRUE(task.validation());

  task.pre_processing();
  task.run();
  task.post_processing();

  // Check the result only on the root process
  if (rank == 0) {
    int expected_count = 0;
    ASSERT_EQ(out[0], expected_count);
  }
}

// Test for a monotonically increasing vector, expecting no violations
TEST(korneeva_e_num_of_orderly_violations_mpi, NoViolations_IncreasingOrder) {
  const int N = 100;  // Size of the vector
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out(1);
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::iota(arr.begin(), arr.end(), 0);  // Fill the vector with increasing numbers (0, 1, 2, ..., N-1)
    data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    data_seq->inputs_count.emplace_back(arr.size());
    data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    data_seq->outputs_count.emplace_back(1);
  }

  korneeva_e_num_of_orderly_violations_mpi::num_of_orderly_violations<int, int> task(data_seq);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  // Check the result only on the root process
  if (world.rank() == 0) {
    int expected_count = 0;  // No violations expected
    ASSERT_EQ(out[0], expected_count);
  }
}

// Test for a monotonically decreasing vector, expecting maximum violations
TEST(korneeva_e_num_of_orderly_violations_mpi, FullViolations_DecreasingOrder) {
  const int N = 100;  // Size of the vector
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out(1);
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();

  // Only the root process initializes the input and output
  if (world.rank() == 0) {
    for (int i = 0; i < N; ++i) {
      arr[i] = N - i;  // Fill the vector with decreasing numbers (N, N-1, N-2, ..., 1)
    }
    data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    data_seq->inputs_count.emplace_back(arr.size());
    data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    data_seq->outputs_count.emplace_back(1);
  }

  korneeva_e_num_of_orderly_violations_mpi::num_of_orderly_violations<int, int> task(data_seq);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  // Check the result only on the root process
  if (world.rank() == 0) {
    int expected_count = N - 1;  // Maximum violations expected
    ASSERT_EQ(out[0], expected_count);
  }
}

// Test for a vector with all equal elements, expecting no violations
TEST(korneeva_e_num_of_orderly_violations_mpi, NoViolations_AllElementsEqual) {
  const int N = 100;
  boost::mpi::communicator world;
  std::vector<int> arr(N, 5);  // All elements equal to 5
  std::vector<int> out(1);
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    data_seq->inputs_count.emplace_back(arr.size());
    data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    data_seq->outputs_count.emplace_back(1);
  }

  korneeva_e_num_of_orderly_violations_mpi::num_of_orderly_violations<int, int> task(data_seq);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  if (world.rank() == 0) {
    int expected_count = 0;  // No violations expected
    ASSERT_EQ(out[0], expected_count);
  }
}

// Test for an empty vector with int data type, expecting no violations
TEST(korneeva_e_num_of_orderly_violations_mpi, NoViolations_EmptyVector_Int) {
  const int N = 0;
  boost::mpi::communicator world;
  std::vector<int> arr(N);  // Empty vector
  std::vector<int> out(1, 0);
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    data_seq->inputs_count.emplace_back(arr.size());
    data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    data_seq->outputs_count.emplace_back(1);
  }

  korneeva_e_num_of_orderly_violations_mpi::num_of_orderly_violations<int, int> task(data_seq);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  if (world.rank() == 0) {
    // For an empty vector, there should be 0 violations
    int expected_count = 0;
    ASSERT_EQ(out[0], expected_count);
  }
}

// Test for a vector of 100 random integers
TEST(korneeva_e_num_of_orderly_violations_mpi, CountViolations_Int_100) {
  const int N = 100;
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out(1);
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::random_device rd;
    std::default_random_engine reng(rd());
    std::uniform_int_distribution<int> dist(0, arr.size());
    std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
    data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    data_seq->inputs_count.emplace_back(arr.size());
    data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    data_seq->outputs_count.emplace_back(1);
  }

  korneeva_e_num_of_orderly_violations_mpi::num_of_orderly_violations<int, int> task(data_seq);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  if (world.rank() == 0) {
    int expected_count = task.count_orderly_violations(arr);  // Calculate expected count
    ASSERT_EQ(out[0], expected_count);                        // Compare with output
  }
}

// Test for a vector of 10,000 random integers
TEST(korneeva_e_num_of_orderly_violations_mpi, CountViolations_Int_10000) {
  const int N = 10000;
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out(1);
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::random_device rd;
    std::default_random_engine reng(rd());
    std::uniform_int_distribution<int> dist(0, arr.size());
    std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
    data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    data_seq->inputs_count.emplace_back(arr.size());
    data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    data_seq->outputs_count.emplace_back(1);
  }

  korneeva_e_num_of_orderly_violations_mpi::num_of_orderly_violations<int, int> task(data_seq);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  if (world.rank() == 0) {
    int expected_count = task.count_orderly_violations(arr);  // Calculate expected count
    ASSERT_EQ(out[0], expected_count);                        // Compare with output
  }
}
// Test for a vector with a number of elements less than the number of processes
TEST(korneeva_e_num_of_orderly_violations_mpi, CountViolations_SmallData_FewerThanProcesses) {
  boost::mpi::communicator world;
  std::vector<int> arr = {3, 2, 1};
  std::vector<int> out(1);
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    data_seq->inputs_count.emplace_back(arr.size());
    data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    data_seq->outputs_count.emplace_back(1);
  }

  korneeva_e_num_of_orderly_violations_mpi::num_of_orderly_violations<int, int> task(data_seq);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  if (world.rank() == 0) {
    int expected_count = task.count_orderly_violations(arr);
    ASSERT_EQ(out[0], expected_count);
  }
}
