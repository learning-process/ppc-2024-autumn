// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <iostream>

#include "mpi/lavrentyev_a_line_topology/include/ops_mpi.hpp"



std::vector<int> lavrentyev_generate_random_vector(size_t size) {
  std::srand(static_cast<unsigned>(std::time(nullptr)));
  std::vector<int> vec(size);
  for (size_t i = 0; i < size; i++) {
    vec[i] = (std::rand() % 2000) - 1000;
  }
  return vec;
}

TEST(lavrentyev_a_line_topology_mpi, MultiProcessCorrectDataTransfer) {
  boost::mpi::communicator world;

  const int start_proc = 0;
  const int end_proc = world.size() - 1;
  const size_t num_elems = 10000;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned int>(start_proc), static_cast<unsigned int>(end_proc),
                             static_cast<unsigned int>(num_elems)};

  std::vector<int> input_data;
  std::vector<int> output_data(num_elems, -1);
  std::vector<int> received_path(end_proc - start_proc + 1, -1);

  if (world.rank() == start_proc) {
    input_data = lavrentyev_generate_random_vector(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  }

  if (world.rank() == end_proc) {
    task_data->outputs = {reinterpret_cast<uint8_t*>(output_data.data()),
                          reinterpret_cast<uint8_t*>(received_path.data())};
    task_data->outputs_count = {static_cast<unsigned int>(output_data.size()),
                                static_cast<unsigned int>(received_path.size())};
  }

  MPI_Barrier(MPI_COMM_WORLD);

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);

  ASSERT_TRUE(task.validation());

  ASSERT_TRUE(task.pre_processing());

  ASSERT_TRUE(task.run());

  ASSERT_TRUE(task.post_processing());

  if (world.rank() == end_proc) {
    ASSERT_EQ(input_data, output_data);
    for (size_t i = 0; i < received_path.size(); ++i) {
      ASSERT_EQ(received_path[i], start_proc + static_cast<int>(i));
    }
  }
}

TEST(lavrentyev_a_line_topology_mpi, ValidationInvalidStartProc) {
  boost::mpi::communicator world;

  int start_proc = -1;
  int end_proc = (world.size() > 1) ? world.size() - 1 : 0;
  int num_elems = 100;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned int>(start_proc), static_cast<unsigned int>(end_proc),
                             static_cast<unsigned int>(num_elems)};

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);

  ASSERT_FALSE(task.validation());
}

TEST(lavrentyev_a_line_topology_mpi, ValidationInvalidEndProc) {
  boost::mpi::communicator world;

  const int start_proc = 0;
  const int end_proc = -1;
  const int num_elems = 100;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned int>(start_proc), static_cast<unsigned int>(end_proc),
                             static_cast<unsigned int>(num_elems)};

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);

  ASSERT_FALSE(task.validation());
}

TEST(lavrentyev_a_line_topology_mpi, ValidationNegativeNumberOfElements) {
  boost::mpi::communicator world;

  const int start_proc = 0;
  const int end_proc = (world.size() > 1) ? world.size() - 1 : 0;
  const int num_elems = -50;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned int>(start_proc), static_cast<unsigned int>(end_proc),
                             static_cast<unsigned int>(num_elems)};

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);

  ASSERT_FALSE(task.validation());
}

TEST(lavrentyev_a_line_topology_mpi, ValidationMissingInputData) {
  boost::mpi::communicator world;

  const int start_proc = 0;
  const int end_proc = (world.size() > 1) ? world.size() - 1 : 0;
  const int num_elems = 1000;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned int>(start_proc), static_cast<unsigned int>(end_proc),
                             static_cast<unsigned int>(num_elems)};

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);

  if (world.rank() == start_proc) {
    if (world.size() == 1) {
      SUCCEED();
    } else {
      ASSERT_FALSE(task.validation());
    }
  } else {
    SUCCEED();
  }
}

TEST(lavrentyev_a_line_topology_mpi, ValidationMissingOutputData) {
  boost::mpi::communicator world;

  const int start_proc = 0;
  const int end_proc = (world.size() > 1) ? world.size() - 1 : 0;
  const int num_elems = 1000;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned int>(start_proc), static_cast<unsigned int>(end_proc),
                             static_cast<unsigned int>(num_elems)};

  if (world.rank() == start_proc) {
    auto input_data = lavrentyev_generate_random_vector(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  }

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);

  if (world.rank() == end_proc) {
    if (world.size() == 1) {
      SUCCEED();
    } else {
      ASSERT_FALSE(task.validation());
    }
  } else {
    SUCCEED();
  }
}

TEST(lavrentyev_a_line_topology_mpi, ValidationInsufficientInputsCount) {
  boost::mpi::communicator world;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {100};

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);

  ASSERT_FALSE(task.validation());
}

TEST(lavrentyev_a_line_topology_mpi, MultiProcessCorrectDataTransfer_1024) {
  boost::mpi::communicator world;

  const int start_proc = 0;
  const int end_proc = world.size() - 1;
  const size_t num_elems = 1024;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned int>(start_proc), static_cast<unsigned int>(end_proc),
                             static_cast<unsigned int>(num_elems)};
  std::vector<int> input_data;
  std::vector<int> output_data(num_elems, -1);
  std::vector<int> received_path(end_proc - start_proc + 1, -1);

  if (world.rank() == start_proc) {
    input_data = lavrentyev_generate_random_vector(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  }

  if (world.rank() == end_proc) {
    task_data->outputs = {reinterpret_cast<uint8_t*>(output_data.data()),
                          reinterpret_cast<uint8_t*>(received_path.data())};
    task_data->outputs_count = {static_cast<unsigned int>(output_data.size()),
                                static_cast<unsigned int>(received_path.size())};
  }

  MPI_Barrier(MPI_COMM_WORLD);

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);
  ASSERT_TRUE(task.validation());

  ASSERT_TRUE(task.pre_processing());

  ASSERT_TRUE(task.run());

  ASSERT_TRUE(task.post_processing());

  if (world.rank() == end_proc) {
    ASSERT_EQ(input_data, output_data);
    for (size_t i = 0; i < received_path.size(); ++i) {
      ASSERT_EQ(received_path[i], start_proc + static_cast<int>(i));
    }
  }
}

TEST(lavrentyev_a_line_topology_mpi, MultiProcessCorrectDataTransfer_2048) {
  boost::mpi::communicator world;

  const int start_proc = 0;
  const int end_proc = world.size() - 1;
  const size_t num_elems = 2048;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned int>(start_proc), static_cast<unsigned int>(end_proc),
                             static_cast<unsigned int>(num_elems)};
  std::vector<int> input_data;
  std::vector<int> output_data(num_elems, -1);
  std::vector<int> received_path(end_proc - start_proc + 1, -1);

  if (world.rank() == start_proc) {
    input_data = lavrentyev_generate_random_vector(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  }

  if (world.rank() == end_proc) {
    task_data->outputs = {reinterpret_cast<uint8_t*>(output_data.data()),
                          reinterpret_cast<uint8_t*>(received_path.data())};
    task_data->outputs_count = {static_cast<unsigned int>(output_data.size()),
                                static_cast<unsigned int>(received_path.size())};
  }

  MPI_Barrier(MPI_COMM_WORLD);

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);
  ASSERT_TRUE(task.validation());

  ASSERT_TRUE(task.pre_processing());

  ASSERT_TRUE(task.run());

  ASSERT_TRUE(task.post_processing());

  if (world.rank() == end_proc) {
    ASSERT_EQ(input_data, output_data);
    for (size_t i = 0; i < received_path.size(); ++i) {
      ASSERT_EQ(received_path[i], start_proc + static_cast<int>(i));
    }
  }
}

TEST(lavrentyev_a_line_topology_mpi, MultiProcessCorrectDataTransfer_4096) {
  boost::mpi::communicator world;

  const int start_proc = 0;
  const int end_proc = world.size() - 1;
  const size_t num_elems = 4096;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned int>(start_proc), static_cast<unsigned int>(end_proc),
                             static_cast<unsigned int>(num_elems)};
  std::vector<int> input_data;
  std::vector<int> output_data(num_elems, -1);
  std::vector<int> received_path(end_proc - start_proc + 1, -1);

  if (world.rank() == start_proc) {
    input_data = lavrentyev_generate_random_vector(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  }

  if (world.rank() == end_proc) {
    task_data->outputs = {reinterpret_cast<uint8_t*>(output_data.data()),
                          reinterpret_cast<uint8_t*>(received_path.data())};
    task_data->outputs_count = {static_cast<unsigned int>(output_data.size()),
                                static_cast<unsigned int>(received_path.size())};
  }

  MPI_Barrier(MPI_COMM_WORLD);

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);
  ASSERT_TRUE(task.validation());

  ASSERT_TRUE(task.pre_processing());

  ASSERT_TRUE(task.run());

  ASSERT_TRUE(task.post_processing());

  if (world.rank() == end_proc) {
    ASSERT_EQ(input_data, output_data);
    for (size_t i = 0; i < received_path.size(); ++i) {
      ASSERT_EQ(received_path[i], start_proc + static_cast<int>(i));
    }
  }
}

TEST(lavrentyev_a_line_topology_mpi, MultiProcessCorrectDataTransfer_8192) {
  boost::mpi::communicator world;

  const int start_proc = 0;
  const int end_proc = world.size() - 1;
  const size_t num_elems = 8192;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned int>(start_proc), static_cast<unsigned int>(end_proc),
                             static_cast<unsigned int>(num_elems)};
  std::vector<int> input_data;
  std::vector<int> output_data(num_elems, -1);
  std::vector<int> received_path(end_proc - start_proc + 1, -1);

  if (world.rank() == start_proc) {
    input_data = lavrentyev_generate_random_vector(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  }

  if (world.rank() == end_proc) {
    task_data->outputs = {reinterpret_cast<uint8_t*>(output_data.data()),
                          reinterpret_cast<uint8_t*>(received_path.data())};
    task_data->outputs_count = {static_cast<unsigned int>(output_data.size()),
                                static_cast<unsigned int>(received_path.size())};
  }

  MPI_Barrier(MPI_COMM_WORLD);

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);
  ASSERT_TRUE(task.validation());

  ASSERT_TRUE(task.pre_processing());

  ASSERT_TRUE(task.run());

  ASSERT_TRUE(task.post_processing());

  if (world.rank() == end_proc) {
    ASSERT_EQ(input_data, output_data);
    for (size_t i = 0; i < received_path.size(); ++i) {
      ASSERT_EQ(received_path[i], start_proc + static_cast<int>(i));
    }
  }
}

TEST(lavrentyev_a_line_topology_mpi, MultiProcessCorrectDataTransfer_16384) {
  boost::mpi::communicator world;

  const int start_proc = 0;
  const int end_proc = world.size() - 1;
  const size_t num_elems = 16384;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned int>(start_proc), static_cast<unsigned int>(end_proc),
                             static_cast<unsigned int>(num_elems)};
  std::vector<int> input_data;
  std::vector<int> output_data(num_elems, -1);
  std::vector<int> received_path(end_proc - start_proc + 1, -1);

  if (world.rank() == start_proc) {
    input_data = lavrentyev_generate_random_vector(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  }

  if (world.rank() == end_proc) {
    task_data->outputs = {reinterpret_cast<uint8_t*>(output_data.data()),
                          reinterpret_cast<uint8_t*>(received_path.data())};
    task_data->outputs_count = {static_cast<unsigned int>(output_data.size()),
                                static_cast<unsigned int>(received_path.size())};
  }

  MPI_Barrier(MPI_COMM_WORLD);

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);
  ASSERT_TRUE(task.validation());

  ASSERT_TRUE(task.pre_processing());

  ASSERT_TRUE(task.run());

  ASSERT_TRUE(task.post_processing());

  if (world.rank() == end_proc) {
    ASSERT_EQ(input_data, output_data);
    for (size_t i = 0; i < received_path.size(); ++i) {
      ASSERT_EQ(received_path[i], start_proc + static_cast<int>(i));
    }
  }
}

TEST(lavrentyev_a_line_topology_mpi, MultiProcessCorrectDataTransfer_2187) {
  boost::mpi::communicator world;

  const int start_proc = 0;
  const int end_proc = world.size() - 1;
  const size_t num_elems = 2187;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned int>(start_proc), static_cast<unsigned int>(end_proc),
                             static_cast<unsigned int>(num_elems)};
  std::vector<int> input_data;
  std::vector<int> output_data(num_elems, -1);
  std::vector<int> received_path(end_proc - start_proc + 1, -1);

  if (world.rank() == start_proc) {
    input_data = lavrentyev_generate_random_vector(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  }

  if (world.rank() == end_proc) {
    task_data->outputs = {reinterpret_cast<uint8_t*>(output_data.data()),
                          reinterpret_cast<uint8_t*>(received_path.data())};
    task_data->outputs_count = {static_cast<unsigned int>(output_data.size()),
                                static_cast<unsigned int>(received_path.size())};
  }

  MPI_Barrier(MPI_COMM_WORLD);

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);
  ASSERT_TRUE(task.validation());

  ASSERT_TRUE(task.pre_processing());

  ASSERT_TRUE(task.run());

  ASSERT_TRUE(task.post_processing());

  if (world.rank() == end_proc) {
    ASSERT_EQ(input_data, output_data);
    for (size_t i = 0; i < received_path.size(); ++i) {
      ASSERT_EQ(received_path[i], start_proc + static_cast<int>(i));
    }
  }
}

TEST(lavrentyev_a_line_topology_mpi, MultiProcessCorrectDataTransfer_6561) {
  boost::mpi::communicator world;

  const int start_proc = 0;
  const int end_proc = world.size() - 1;
  const size_t num_elems = 2048;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned int>(start_proc), static_cast<unsigned int>(end_proc),
                             static_cast<unsigned int>(num_elems)};
  std::vector<int> input_data;
  std::vector<int> output_data(num_elems, -1);
  std::vector<int> received_path(end_proc - start_proc + 1, -1);

  if (world.rank() == start_proc) {
    input_data = lavrentyev_generate_random_vector(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  }

  if (world.rank() == end_proc) {
    task_data->outputs = {reinterpret_cast<uint8_t*>(output_data.data()),
                          reinterpret_cast<uint8_t*>(received_path.data())};
    task_data->outputs_count = {static_cast<unsigned int>(output_data.size()),
                                static_cast<unsigned int>(received_path.size())};
  }

  MPI_Barrier(MPI_COMM_WORLD);

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);
  ASSERT_TRUE(task.validation());

  ASSERT_TRUE(task.pre_processing());

  ASSERT_TRUE(task.run());

  ASSERT_TRUE(task.post_processing());

  if (world.rank() == end_proc) {
    ASSERT_EQ(input_data, output_data);
    for (size_t i = 0; i < received_path.size(); ++i) {
      ASSERT_EQ(received_path[i], start_proc + static_cast<int>(i));
    }
  }
}

TEST(lavrentyev_a_line_topology_mpi, MultiProcessCorrectDataTransfer_19638) {
  boost::mpi::communicator world;

  const int start_proc = 0;
  const int end_proc = world.size() - 1;
  const size_t num_elems = 2048;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned int>(start_proc), static_cast<unsigned int>(end_proc),
                             static_cast<unsigned int>(num_elems)};
  std::vector<int> input_data;
  std::vector<int> output_data(num_elems, -1);
  std::vector<int> received_path(end_proc - start_proc + 1, -1);

  if (world.rank() == start_proc) {
    input_data = lavrentyev_generate_random_vector(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  }

  if (world.rank() == end_proc) {
    task_data->outputs = {reinterpret_cast<uint8_t*>(output_data.data()),
                          reinterpret_cast<uint8_t*>(received_path.data())};
    task_data->outputs_count = {static_cast<unsigned int>(output_data.size()),
                                static_cast<unsigned int>(received_path.size())};
  }

  MPI_Barrier(MPI_COMM_WORLD);

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);
  ASSERT_TRUE(task.validation());

  ASSERT_TRUE(task.pre_processing());

  ASSERT_TRUE(task.run());

  ASSERT_TRUE(task.post_processing());

  if (world.rank() == end_proc) {
    ASSERT_EQ(input_data, output_data);
    for (size_t i = 0; i < received_path.size(); ++i) {
      ASSERT_EQ(received_path[i], start_proc + static_cast<int>(i));
    }
  }
}

TEST(lavrentyev_a_line_topology_mpi, MultiProcessCorrectDataTransfer_2791) {
  boost::mpi::communicator world;

  const int start_proc = 0;
  const int end_proc = world.size() - 1;
  const size_t num_elems = 2791;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned int>(start_proc), static_cast<unsigned int>(end_proc),
                             static_cast<unsigned int>(num_elems)};
  std::vector<int> input_data;
  std::vector<int> output_data(num_elems, -1);
  std::vector<int> received_path(end_proc - start_proc + 1, -1);

  if (world.rank() == start_proc) {
    input_data = lavrentyev_generate_random_vector(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  }

  if (world.rank() == end_proc) {
    task_data->outputs = {reinterpret_cast<uint8_t*>(output_data.data()),
                          reinterpret_cast<uint8_t*>(received_path.data())};
    task_data->outputs_count = {static_cast<unsigned int>(output_data.size()),
                                static_cast<unsigned int>(received_path.size())};
  }

  MPI_Barrier(MPI_COMM_WORLD);

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);
  ASSERT_TRUE(task.validation());

  ASSERT_TRUE(task.pre_processing());

  ASSERT_TRUE(task.run());

  ASSERT_TRUE(task.post_processing());

  if (world.rank() == end_proc) {
    ASSERT_EQ(input_data, output_data);
    for (size_t i = 0; i < received_path.size(); ++i) {
      ASSERT_EQ(received_path[i], start_proc + static_cast<int>(i));
    }
  }
}

TEST(lavrentyev_a_line_topology_mpi, MultiProcessCorrectDataTransfer_5021) {
  boost::mpi::communicator world;

  const int start_proc = 0;
  const int end_proc = world.size() - 1;
  const size_t num_elems = 5021;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned int>(start_proc), static_cast<unsigned int>(end_proc),
                             static_cast<unsigned int>(num_elems)};
  std::vector<int> input_data;
  std::vector<int> output_data(num_elems, -1);
  std::vector<int> received_path(end_proc - start_proc + 1, -1);

  if (world.rank() == start_proc) {
    input_data = lavrentyev_generate_random_vector(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  }

  if (world.rank() == end_proc) {
    task_data->outputs = {reinterpret_cast<uint8_t*>(output_data.data()),
                          reinterpret_cast<uint8_t*>(received_path.data())};
    task_data->outputs_count = {static_cast<unsigned int>(output_data.size()),
                                static_cast<unsigned int>(received_path.size())};
  }

  MPI_Barrier(MPI_COMM_WORLD);

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);
  ASSERT_TRUE(task.validation());

  ASSERT_TRUE(task.pre_processing());

  ASSERT_TRUE(task.run());

  ASSERT_TRUE(task.post_processing());

  if (world.rank() == end_proc) {
    ASSERT_EQ(input_data, output_data);
    for (size_t i = 0; i < received_path.size(); ++i) {
      ASSERT_EQ(received_path[i], start_proc + static_cast<int>(i));
    }
  }
}

TEST(lavrentyev_a_line_topology_mpi, MultiProcessCorrectDataTransfer_7517) {
  boost::mpi::communicator world;

  const int start_proc = 0;
  const int end_proc = world.size() - 1;
  const size_t num_elems = 7517;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned int>(start_proc), static_cast<unsigned int>(end_proc),
                             static_cast<unsigned int>(num_elems)};
  std::vector<int> input_data;
  std::vector<int> output_data(num_elems, -1);
  std::vector<int> received_path(end_proc - start_proc + 1, -1);

  if (world.rank() == start_proc) {
    input_data = lavrentyev_generate_random_vector(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  }

  if (world.rank() == end_proc) {
    task_data->outputs = {reinterpret_cast<uint8_t*>(output_data.data()),
                          reinterpret_cast<uint8_t*>(received_path.data())};
    task_data->outputs_count = {static_cast<unsigned int>(output_data.size()),
                                static_cast<unsigned int>(received_path.size())};
  }

  MPI_Barrier(MPI_COMM_WORLD);

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);
  ASSERT_TRUE(task.validation());

  ASSERT_TRUE(task.pre_processing());

  ASSERT_TRUE(task.run());

  ASSERT_TRUE(task.post_processing());

  if (world.rank() == end_proc) {
    ASSERT_EQ(input_data, output_data);
    for (size_t i = 0; i < received_path.size(); ++i) {
      ASSERT_EQ(received_path[i], start_proc + static_cast<int>(i));
    }
  }
}
