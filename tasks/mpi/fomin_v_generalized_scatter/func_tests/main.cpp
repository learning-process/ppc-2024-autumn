#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/fomin_v_generalized_scatter/include/ops_mpi.hpp"

TEST(fomin_v_generalized_scatter, ScatterIntegers) {
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  int root = 0;
  const int recvcount = 10;
  const int data_size = size * recvcount;
  int* sendbuf = nullptr;
  auto* recvbuf = new int[recvcount];

  if (rank == root) {
    sendbuf = new int[data_size];
    for (int i = 0; i < data_size; ++i) {
      sendbuf[i] = i;
    }
  }

  fomin_v_generalized_scatter::generalized_scatter(sendbuf, data_size, MPI_INT, recvbuf, recvcount, MPI_INT, root,
                                                   MPI_COMM_WORLD);

  // Generate pre-order rank order
  std::vector<int> pre_order_rank;
  fomin_v_generalized_scatter::pre_order(0, size, pre_order_rank);

  // Find the position of the current rank in pre-order rank order
  auto it = std::find(pre_order_rank.begin(), pre_order_rank.end(), rank);
  int position = std::distance(pre_order_rank.begin(), it);

  // Set expected starting index
  int expected_start = position * recvcount;

  // Set expected values
  int expected[recvcount];
  for (int i = 0; i < recvcount; ++i) {
    expected[i] = expected_start + i;
  }

  // Verify received data against expected data
  for (int i = 0; i < recvcount; ++i) {
    EXPECT_EQ(recvbuf[i], expected[i]);
  }

  delete[] sendbuf;
  delete[] recvbuf;
}

TEST(fomin_v_generalized_scatter, ScatterFloats) {
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  int root = 0;
  const int recvcount = 10;
  const int data_size = size * recvcount;
  float* sendbuf = nullptr;
  auto* recvbuf = new float[recvcount];

  if (rank == root) {
    sendbuf = new float[data_size];
    for (int i = 0; i < data_size; ++i) {
      sendbuf[i] = static_cast<float>(i);
    }
  }

  fomin_v_generalized_scatter::generalized_scatter(sendbuf, data_size, MPI_FLOAT, recvbuf, recvcount, MPI_FLOAT, root,
                                                   MPI_COMM_WORLD);

  // Generate pre-order traversal rank order
  std::vector<int> pre_order_rank;
  fomin_v_generalized_scatter::pre_order(0, size, pre_order_rank);

  // Find the position of the current rank in pre-order rank order
  auto it = std::find(pre_order_rank.begin(), pre_order_rank.end(), rank);
  int position = std::distance(pre_order_rank.begin(), it);

  // Set expected starting index
  int expected_start = position * recvcount;

  // Set expected values
  float expected[recvcount];
  for (int i = 0; i < recvcount; ++i) {
    expected[i] = static_cast<float>(expected_start + i);
  }

  // Verify received data against expected data
  for (int i = 0; i < recvcount; ++i) {
    EXPECT_NEAR(recvbuf[i], expected[i], 1e-5);
  }

  delete[] sendbuf;
  delete[] recvbuf;
}

TEST(fomin_v_generalized_scatter, ScatterDoubles) {
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  int root = 0;
  const int recvcount = 10;
  const int data_size = size * recvcount;
  double* sendbuf = nullptr;
  auto* recvbuf = new double[recvcount];

  if (rank == root) {
    sendbuf = new double[data_size];
    for (int i = 0; i < data_size; ++i) {
      sendbuf[i] = static_cast<double>(i);
    }
  }

  fomin_v_generalized_scatter::generalized_scatter(sendbuf, data_size, MPI_DOUBLE, recvbuf, recvcount, MPI_DOUBLE, root,
                                                   MPI_COMM_WORLD);

  // Generate pre-order traversal rank order
  std::vector<int> pre_order_rank;
  fomin_v_generalized_scatter::pre_order(0, size, pre_order_rank);

  // Find the position of the current rank in pre-order rank order
  auto it = std::find(pre_order_rank.begin(), pre_order_rank.end(), rank);
  int position = std::distance(pre_order_rank.begin(), it);

  // Set expected starting index
  int expected_start = position * recvcount;

  // Set expected values
  double expected[recvcount];
  for (int i = 0; i < recvcount; ++i) {
    expected[i] = static_cast<double>(expected_start + i);
  }

  // Verify received data against expected data
  for (int i = 0; i < recvcount; ++i) {
    EXPECT_DOUBLE_EQ(recvbuf[i], expected[i]);
  }

  delete[] sendbuf;
  delete[] recvbuf;
}

TEST(fomin_v_generalized_scatter, NonPowerOfTwoProcesses) {
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  int root = 0;
  const int recvcount = 10;
  const int data_size = size * recvcount;
  int* sendbuf = nullptr;
  auto* recvbuf = new int[recvcount];

  if (rank == root) {
    sendbuf = new int[data_size];
    for (int i = 0; i < data_size; ++i) {
      sendbuf[i] = i;
    }
  }

  fomin_v_generalized_scatter::generalized_scatter(sendbuf, data_size, MPI_INT, recvbuf, 10, MPI_INT, root,
                                                   MPI_COMM_WORLD);

  // Generate pre-order traversal rank order
  std::vector<int> pre_order_rank;
  fomin_v_generalized_scatter::pre_order(0, size, pre_order_rank);

  // Find the position of the current rank in pre_order_rank
  auto it = std::find(pre_order_rank.begin(), pre_order_rank.end(), rank);
  int position = std::distance(pre_order_rank.begin(), it);

  // Set expected starting index
  int expected_start = position * recvcount;

  // Set expected values
  int expected[recvcount];
  for (int i = 0; i < recvcount; ++i) {
    expected[i] = expected_start + i;
  }

  // Verify received data against expected data
  for (int i = 0; i < recvcount; ++i) {
    EXPECT_EQ(recvbuf[i], expected[i]);
  }

  delete[] sendbuf;
  delete[] recvbuf;
}

TEST(fomin_v_generalized_scatter, ZeroElementsScatter) {
  boost::mpi::communicator world;
  int root = 0;
  const int data_size = 0;
  int* sendbuf = nullptr;
  int* recvbuf = nullptr;
  int result = fomin_v_generalized_scatter::generalized_scatter(sendbuf, data_size, MPI_INT, recvbuf, 0, MPI_INT, root,
                                                                MPI_COMM_WORLD);

  // Check that the function returns MPI_SUCCESS
  EXPECT_EQ(result, MPI_SUCCESS);
}