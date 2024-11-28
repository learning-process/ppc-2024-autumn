#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/sidorina_p_broadcast/include/ops_mpi.hpp"

TEST(sidorina_p_broadcast_mpi, Test_array3_term2_ref) {
  boost::mpi::communicator world;

  std::vector<int> reference;
  std::vector<int> array;
  std::vector<int> terms;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int sz1 = 3;
    int sz2 = 2;
    array = std::vector<int>({1, 2, 3});
    terms = std::vector<int>({1, 2});
    reference.resize(sz1);
    result.resize(sz1);
    for (int i = 0; i < sz1; i++) {
      for (int j = 0; j < sz2; j++) {
        int a = array[i] + terms[j];
        reference[i] += a;
      }
    }

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataGlob->inputs_count.emplace_back(array.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskDataGlob->inputs_count.emplace_back(terms.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataGlob->outputs_count.emplace_back(result.size());
  }

  sidorina_p_broadcast_mpi::Broadcast testMpiTaskParallel(taskDataGlob);
  testMpiTaskParallel.broadcast_fn = [](const boost::mpi::communicator& comm, int* values, int n, int root) {
    boost::mpi::broadcast(comm, values, n, root);
  };
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(reference, result);
  }
}

/* TEST(sidorina_p_broadcast_mpi, Test_validation_array_1) {
  boost::mpi::communicator world;

  std::vector<int> array;
  std::vector<int> terms;
  std::vector<int> m_result;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    array = std::vector<int>();
    terms = std::vector<int>({1, 2, 3});

    m_result.resize(array.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataGlob->inputs_count.emplace_back(array.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskDataGlob->inputs_count.emplace_back(terms.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_result.data()));
    taskDataGlob->outputs_count.emplace_back(m_result.size());
  }

  sidorina_p_broadcast_mpi::Broadcast testMpiTaskParallel(taskDataGlob);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
}

TEST(sidorina_p_broadcast_mpi, Test_validation_terms_1)*/ {
  boost::mpi::communicator world;

  std::vector<int> array;
  std::vector<int> terms;
  std::vector<int> m_result;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    array = std::vector<int>({1, 2, 3});
    terms = std::vector<int>();

    m_result.resize(array.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataGlob->inputs_count.emplace_back(array.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskDataGlob->inputs_count.emplace_back(terms.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_result.data()));
    taskDataGlob->outputs_count.emplace_back(m_result.size());
  }

  sidorina_p_broadcast_mpi::Broadcast testMpiTaskParallel(taskDataGlob);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
}