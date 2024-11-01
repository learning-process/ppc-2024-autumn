// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/kozlova_e_lexic_order/include/ops_mpi.hpp"

TEST(kozlova_e_lexic_order, Test_mixed_strings) {
  boost::mpi::communicator world;
  std::vector<std::string> input_strings = {"aBcdef", "cDefga"};
  std::vector<int> resMPI(2, 0);
  std::vector<int> answer = {1, 0};
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (const auto& str : input_strings) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *> (str.c_str())));
    }
    taskDataPar->inputs_count.emplace_back(2);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  kozlova_e_lexic_order_mpi::StringComparatorMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> resSeq(2, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (const auto &str : input_strings) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(str.c_str())));
    }
    taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(2));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(static_cast<uint32_t>(2));

    kozlova_e_lexic_order_mpi::StringComparatorSeq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(resMPI, answer);
  }
}
TEST(kozlova_e_lexic_order, Test_Ordered_Strings) {
  boost::mpi::communicator world;
  std::vector<std::string> input_strings = {"abcde", "abcdef"};
  std::vector<int> resMPI(2, 0);
  std::vector<int> answer(2, 1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (const auto &str : input_strings) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(str.c_str())));
    }
    taskDataPar->inputs_count.emplace_back(2);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  kozlova_e_lexic_order_mpi::StringComparatorMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> resSeq(2, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (const auto &str : input_strings) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(str.c_str())));
    }
    taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(2));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(static_cast<uint32_t>(2));

    kozlova_e_lexic_order_mpi::StringComparatorSeq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(resMPI, resSeq);
  }
}

TEST(kozlova_e_lexic_order, Test_Unordered_Strings) {
  boost::mpi::communicator world;
  std::vector<std::string> input_strings = {"cba", "fedcba"};
  std::vector<int> resMPI(2, 0);
  std::vector<int> answer(2,0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (const auto &str : input_strings) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(str.c_str())));
    }
    taskDataPar->inputs_count.emplace_back(2);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  kozlova_e_lexic_order_mpi::StringComparatorMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> resSeq(2, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (const auto &str : input_strings) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(str.c_str())));
    }
    taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(2));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(static_cast<uint32_t>(2));

    kozlova_e_lexic_order_mpi::StringComparatorSeq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(resMPI, resSeq);
  }
}

TEST(kozlova_e_lexic_order, Test_Boundary_Case) {
  boost::mpi::communicator world;
  std::vector<std::string> input_strings = {"a", "b"};
  std::vector<int> resMPI(2, 0);
  std::vector<int> answer(2, 1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (const auto &str : input_strings) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(str.c_str())));
    }
    taskDataPar->inputs_count.emplace_back(2);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  kozlova_e_lexic_order_mpi::StringComparatorMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(resMPI, answer);
  }
}

TEST(kozlova_e_lexic_order, Test_empty_strings) {
  boost::mpi::communicator world;
  std::vector<std::string> input_strings = {"", ""};
  std::vector<int> resMPI(2, 0);
  std::vector<int> answer(2, 1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (const auto &str : input_strings) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(str.c_str())));
    }
    taskDataPar->inputs_count.emplace_back(2);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  kozlova_e_lexic_order_mpi::StringComparatorMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> resSeq(2, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (const auto &str : input_strings) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(str.c_str())));
    }
    taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(2));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(static_cast<uint32_t>(2));

    kozlova_e_lexic_order_mpi::StringComparatorSeq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(resMPI, resSeq);
  }
}
