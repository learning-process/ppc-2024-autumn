#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <cstdlib>
#include <ctime>
#include <vector>

#include "mpi/lavrentyev_a_line_topology/include/ops_mpi.hpp"

std::vector<int> lavrentyrev_generate_random_vector(size_t size) {
  std::srand(static_cast<unsigned>(std::time(nullptr)));
  std::vector<int> vec(size);
  for (size_t i = 0; i < size; i++) {
    vec[i] = std::rand() % 1000;
  }
  return vec;
}

TEST(lavrentyev_a_line_topology_mpi, MultiProcessCorrectDataTransfer) {
  boost::mpi::communicator world;

  const size_t start_proc = 0;
  const size_t end_proc = world.size() > 1 ? static_cast<size_t>(world.size() - 1) : 0;
  const size_t num_elems = 10000;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned int>(start_proc), static_cast<unsigned int>(end_proc),
                             static_cast<unsigned int>(num_elems)};

  std::vector<int> input_data;
  std::vector<int> output_data(num_elems, -1);
  std::vector<int> received_path(end_proc - start_proc + 1, -1);

  if (static_cast<size_t>(world.rank()) == start_proc) {
    input_data = lavrentyrev_generate_random_vector(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));

    if (start_proc != end_proc) {
      world.send(end_proc, 0, input_data);
    }
  }

  if (static_cast<size_t>(world.rank()) == end_proc) {
    task_data->outputs = {reinterpret_cast<uint8_t*>(output_data.data()),
                          reinterpret_cast<uint8_t*>(received_path.data())};
    task_data->outputs_count = {static_cast<unsigned int>(output_data.size()),
                                static_cast<unsigned int>(received_path.size())};
    if (start_proc != end_proc) {
      world.recv(start_proc, 0, input_data);
    }
  }

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  if (static_cast<size_t>(world.rank()) == end_proc) {
    for (size_t i = 0; i < output_data.size(); i++) {
      ASSERT_EQ(input_data[i], output_data[i]);
    }
    for (size_t i = 0; i < received_path.size(); ++i) {
      ASSERT_EQ(received_path[i], static_cast<int>(start_proc) + static_cast<int>(i));
    }
  }
}

TEST(lavrentyev_a_line_topology_mpi, ValidationInvalidStartProc) {
  boost::mpi::communicator world;

  auto start_proc = static_cast<unsigned int>(static_cast<size_t>(world.size()));
  auto end_proc = world.size() > 1 ? static_cast<unsigned int>(static_cast<size_t>(world.size() - 1)) : 0;
  auto num_elems = 100u;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {start_proc, end_proc, num_elems};

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);

  ASSERT_FALSE(task.validation());
}

TEST(lavrentyev_a_line_topology_mpi, ValidationInvalidDestination) {
  boost::mpi::communicator world;

  auto start_proc = 0u;
  auto end_proc = static_cast<unsigned int>(static_cast<size_t>(world.size()));
  auto num_elems = 100u;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {start_proc, end_proc, num_elems};

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);

  ASSERT_FALSE(task.validation());
}

TEST(lavrentyev_a_line_topology_mpi, ValidationNegativeNumberOfElements) {
  boost::mpi::communicator world;

  unsigned start_proc = 0;
  unsigned end_proc = (world.size() > 1) ? world.size() - 1 : 0;
  int num_elems = -50;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {start_proc, end_proc, static_cast<unsigned int>(num_elems)};

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);

  ASSERT_FALSE(task.validation());
}

TEST(lavrentyev_a_line_topology_mpi, ValidationMissingInputData) {
  boost::mpi::communicator world;

  const unsigned start_proc = 0;
  const unsigned end_proc = (world.size() > 1) ? world.size() - 1 : 0;
  const unsigned num_elems = 1000;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {start_proc, end_proc, num_elems};

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);

  if (world.rank() == start_proc) {
    ASSERT_FALSE(task.validation());
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
  task_data->inputs_count = {static_cast<unsigned>(start_proc), static_cast<unsigned>(end_proc),
                             static_cast<unsigned>(num_elems)};

  if (world.rank() == start_proc) {
    auto input_data = lavrentyrev_generate_random_vector(static_cast<size_t>(num_elems));
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  }

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);

  if (world.rank() == end_proc) {
    ASSERT_FALSE(task.validation());
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

  const size_t start_proc = 0;
  const size_t end_proc = world.size() > 1 ? static_cast<size_t>(world.size() - 1) : 0;
  const size_t num_elems = 1024;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned int>(start_proc), static_cast<unsigned int>(end_proc),
                             static_cast<unsigned int>(num_elems)};

  std::vector<int> input_data;
  std::vector<int> output_data(num_elems, -1);
  std::vector<int> received_path(end_proc - start_proc + 1, -1);

  if (static_cast<size_t>(world.rank()) == start_proc) {
    input_data = lavrentyrev_generate_random_vector(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));

    if (start_proc != end_proc) {
      world.send(end_proc, 0, input_data);
    }
  }

  if (static_cast<size_t>(world.rank()) == end_proc) {
    task_data->outputs = {reinterpret_cast<uint8_t*>(output_data.data()),
                          reinterpret_cast<uint8_t*>(received_path.data())};
    task_data->outputs_count = {static_cast<unsigned int>(output_data.size()),
                                static_cast<unsigned int>(received_path.size())};
    if (start_proc != end_proc) {
      world.recv(start_proc, 0, input_data);
    }
  }

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  if (static_cast<size_t>(world.rank()) == end_proc) {
    for (size_t i = 0; i < output_data.size(); i++) {
      ASSERT_EQ(input_data[i], output_data[i]);
    }
    for (size_t i = 0; i < received_path.size(); ++i) {
      ASSERT_EQ(received_path[i], static_cast<int>(start_proc) + static_cast<int>(i));
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
  std::vector<int> received_path(static_cast<size_t>(end_proc - start_proc + 1), -1);

  if (world.rank() == start_proc) {
    input_data = lavrentyrev_generate_random_vector(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data->inputs_count.push_back(static_cast<unsigned int>(input_data.size()));
    if (start_proc != end_proc) {
      world.send(end_proc, 0, input_data);
    }
  }

  if (world.rank() == end_proc) {
    task_data->outputs = {reinterpret_cast<uint8_t*>(output_data.data()),
                          reinterpret_cast<uint8_t*>(received_path.data())};
    task_data->outputs_count = {static_cast<unsigned int>(output_data.size()),
                                static_cast<unsigned int>(received_path.size())};
    if (start_proc != end_proc) {
      world.recv(start_proc, 0, input_data);
    }
  }

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  if (world.rank() == end_proc) {
    for (size_t i = 0; i < output_data.size(); i++) {
      ASSERT_EQ(input_data[i], output_data[i]);
    }
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
  std::vector<int> received_path(static_cast<size_t>(end_proc - start_proc + 1), -1);

  if (world.rank() == start_proc) {
    input_data = lavrentyrev_generate_random_vector(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data->inputs_count.push_back(static_cast<unsigned int>(input_data.size()));
    if (start_proc != end_proc) {
      world.send(end_proc, 0, input_data);
    }
  }

  if (world.rank() == end_proc) {
    task_data->outputs = {reinterpret_cast<uint8_t*>(output_data.data()),
                          reinterpret_cast<uint8_t*>(received_path.data())};
    task_data->outputs_count = {static_cast<unsigned int>(output_data.size()),
                                static_cast<unsigned int>(received_path.size())};
    if (start_proc != end_proc) {
      world.recv(start_proc, 0, input_data);
    }
  }

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  if (world.rank() == end_proc) {
    for (size_t i = 0; i < output_data.size(); i++) {
      ASSERT_EQ(input_data[i], output_data[i]);
    }
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
  std::vector<int> received_path(static_cast<size_t>(end_proc - start_proc + 1), -1);

  if (world.rank() == start_proc) {
    input_data = lavrentyrev_generate_random_vector(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data->inputs_count.push_back(static_cast<unsigned int>(input_data.size()));
    if (start_proc != end_proc) {
      world.send(end_proc, 0, input_data);
    }
  }

  if (world.rank() == end_proc) {
    task_data->outputs = {reinterpret_cast<uint8_t*>(output_data.data()),
                          reinterpret_cast<uint8_t*>(received_path.data())};
    task_data->outputs_count = {static_cast<unsigned int>(output_data.size()),
                                static_cast<unsigned int>(received_path.size())};
    if (start_proc != end_proc) {
      world.recv(start_proc, 0, input_data);
    }
  }

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  if (world.rank() == end_proc) {
    for (size_t i = 0; i < output_data.size(); i++) {
      ASSERT_EQ(input_data[i], output_data[i]);
    }
    for (size_t i = 0; i < received_path.size(); ++i) {
      ASSERT_EQ(received_path[i], start_proc + static_cast<int>(i));
    }
  }
}

TEST(lavrentyev_a_line_topology_mpi, MultiProcessCorrectDataTransfer_19638) {
  boost::mpi::communicator world;

  const int start_proc = 0;
  const int end_proc = world.size() - 1;
  const size_t num_elems = 19638;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned int>(start_proc), static_cast<unsigned int>(end_proc),
                             static_cast<unsigned int>(num_elems)};

  std::vector<int> input_data;
  std::vector<int> output_data(num_elems, -1);
  std::vector<int> received_path(static_cast<size_t>(end_proc - start_proc + 1), -1);

  if (world.rank() == start_proc) {
    input_data = lavrentyrev_generate_random_vector(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data->inputs_count.push_back(static_cast<unsigned int>(input_data.size()));
    if (start_proc != end_proc) {
      world.send(end_proc, 0, input_data);
    }
  }

  if (world.rank() == end_proc) {
    task_data->outputs = {reinterpret_cast<uint8_t*>(output_data.data()),
                          reinterpret_cast<uint8_t*>(received_path.data())};
    task_data->outputs_count = {static_cast<unsigned int>(output_data.size()),
                                static_cast<unsigned int>(received_path.size())};
    if (start_proc != end_proc) {
      world.recv(start_proc, 0, input_data);
    }
  }

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  if (world.rank() == end_proc) {
    for (size_t i = 0; i < output_data.size(); i++) {
      ASSERT_EQ(input_data[i], output_data[i]);
    }
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
  std::vector<int> received_path(static_cast<size_t>(end_proc - start_proc + 1), -1);

  if (world.rank() == start_proc) {
    input_data = lavrentyrev_generate_random_vector(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data->inputs_count.push_back(static_cast<unsigned int>(input_data.size()));
    if (start_proc != end_proc) {
      world.send(end_proc, 0, input_data);
    }
  }

  if (world.rank() == end_proc) {
    task_data->outputs = {reinterpret_cast<uint8_t*>(output_data.data()),
                          reinterpret_cast<uint8_t*>(received_path.data())};
    task_data->outputs_count = {static_cast<unsigned int>(output_data.size()),
                                static_cast<unsigned int>(received_path.size())};
    if (start_proc != end_proc) {
      world.recv(start_proc, 0, input_data);
    }
  }

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  if (world.rank() == end_proc) {
    for (size_t i = 0; i < output_data.size(); i++) {
      ASSERT_EQ(input_data[i], output_data[i]);
    }
    for (size_t i = 0; i < received_path.size(); ++i) {
      ASSERT_EQ(received_path[i], start_proc + static_cast<int>(i));
    }
  }
}

TEST(lavrentyev_a_line_topology_mpi, MultiProcessCorrectDataTransfer_7517) {
  boost::mpi::communicator world;

  const int start_proc = 0;
  const int end_proc = world.size() - 1;
  const size_t num_elems = 1024;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count = {static_cast<unsigned int>(start_proc), static_cast<unsigned int>(end_proc),
                             static_cast<unsigned int>(num_elems)};

  std::vector<int> input_data;
  std::vector<int> output_data(num_elems, -1);
  std::vector<int> received_path(static_cast<size_t>(end_proc - start_proc + 1), -1);

  if (world.rank() == start_proc) {
    input_data = lavrentyrev_generate_random_vector(num_elems);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data->inputs_count.push_back(static_cast<unsigned int>(input_data.size()));
    if (start_proc != end_proc) {
      world.send(end_proc, 0, input_data);
    }
  }

  if (world.rank() == end_proc) {
    task_data->outputs = {reinterpret_cast<uint8_t*>(output_data.data()),
                          reinterpret_cast<uint8_t*>(received_path.data())};
    task_data->outputs_count = {static_cast<unsigned int>(output_data.size()),
                                static_cast<unsigned int>(received_path.size())};
    if (start_proc != end_proc) {
      world.recv(start_proc, 0, input_data);
    }
  }

  lavrentyev_a_line_topology_mpi::TestMPITaskParallel task(task_data);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  if (world.rank() == end_proc) {
    for (size_t i = 0; i < output_data.size(); i++) {
      ASSERT_EQ(input_data[i], output_data[i]);
    }
    for (size_t i = 0; i < received_path.size(); ++i) {
      ASSERT_EQ(received_path[i], start_proc + static_cast<int>(i));
    }
  }
}
