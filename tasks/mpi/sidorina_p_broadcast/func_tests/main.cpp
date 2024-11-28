#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/sidorina_p_broadcast/include/ops_mpi.hpp"

TEST(sidorina_p_broadcast_mpi, Test_arr3_term2_m) {
  boost::mpi::communicator world;

  std::vector<int> array;
  std::vector<int> terms;
  std::vector<int> m_result;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    array = std::vector<int>({1, 2, 3});
    terms = std::vector<int>({1, 2});

    m_result.resize(array.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataGlob->inputs_count.emplace_back(array.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskDataGlob->inputs_count.emplace_back(terms.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_result.data()));
    taskDataGlob->outputs_count.emplace_back(m_result.size());
  }

  sidorina_p_broadcast_mpi::Broadcast testMpiTaskParallel(taskDataGlob);
  testMpiTaskParallel.broadcast_fn = [](const boost::mpi::communicator& comm, int* values, int n, int root) {
    sidorina_p_broadcast_mpi::Broadcast::broadcast_m(comm, values, n, root);
  };
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    result.resize(array.size(), 0);

    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataRef->inputs_count.emplace_back(array.size());
    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskDataRef->inputs_count.emplace_back(terms.size());
    taskDataRef->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataRef->outputs_count.emplace_back(result.size());
  }
  sidorina_p_broadcast_mpi::Broadcast testMpiTaskSequential(taskDataRef);
  testMpiTaskSequential.broadcast_fn = [](const boost::mpi::communicator& comm, int* values, int n, int root) {
    boost::mpi::broadcast(comm, values, n, root);
  };
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(m_result, result);
  }
}

TEST(sidorina_p_broadcast_mpi, Test_array3_term2_ref_m) {
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
    sidorina_p_broadcast_mpi::Broadcast::broadcast_m(comm, values, n, root);
  };
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(reference, result);
  }
}

TEST(sidorina_p_broadcast_mpi, Test_arr3_term3_m) {
  boost::mpi::communicator world;

  std::vector<int> array;
  std::vector<int> terms;
  std::vector<int> m_result;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    array = std::vector<int>({1, 2, 3});
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
  testMpiTaskParallel.broadcast_fn = [](const boost::mpi::communicator& comm, int* values, int n, int root) {
    sidorina_p_broadcast_mpi::Broadcast::broadcast_m(comm, values, n, root);
  };
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    result.resize(array.size(), 0);

    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataRef->inputs_count.emplace_back(array.size());
    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskDataRef->inputs_count.emplace_back(terms.size());
    taskDataRef->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataRef->outputs_count.emplace_back(result.size());
  }
  sidorina_p_broadcast_mpi::Broadcast testMpiTaskSequential(taskDataRef);
  testMpiTaskSequential.broadcast_fn = [](const boost::mpi::communicator& comm, int* values, int n, int root) {
    boost::mpi::broadcast(comm, values, n, root);
  };
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(m_result, result);
  }
}

TEST(sidorina_p_broadcast_mpi, Test_arr3_term6_m) {
  boost::mpi::communicator world;

  std::vector<int> array;
  std::vector<int> terms;
  std::vector<int> m_result;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    array = std::vector<int>({1, 2, 3});
    terms = std::vector<int>({1, 2, 3, 4, 5, 6});

    m_result.resize(array.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataGlob->inputs_count.emplace_back(array.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskDataGlob->inputs_count.emplace_back(terms.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_result.data()));
    taskDataGlob->outputs_count.emplace_back(m_result.size());
  }

  sidorina_p_broadcast_mpi::Broadcast testMpiTaskParallel(taskDataGlob);
  testMpiTaskParallel.broadcast_fn = [](const boost::mpi::communicator& comm, int* values, int n, int root) {
    sidorina_p_broadcast_mpi::Broadcast::broadcast_m(comm, values, n, root);
  };
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    result.resize(array.size(), 0);

    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataRef->inputs_count.emplace_back(array.size());
    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskDataRef->inputs_count.emplace_back(terms.size());
    taskDataRef->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataRef->outputs_count.emplace_back(result.size());
  }
  sidorina_p_broadcast_mpi::Broadcast testMpiTaskSequential(taskDataRef);
  testMpiTaskSequential.broadcast_fn = [](const boost::mpi::communicator& comm, int* values, int n, int root) {
    boost::mpi::broadcast(comm, values, n, root);
  };
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(m_result, result);
  }
}

TEST(sidorina_p_broadcast_mpi, Test_negative_m) {
  boost::mpi::communicator world;

  std::vector<int> array;
  std::vector<int> terms;
  std::vector<int> m_result;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    array = std::vector<int>({1, 2, 3});
    terms = std::vector<int>({-1, 2, -3});

    m_result.resize(array.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataGlob->inputs_count.emplace_back(array.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskDataGlob->inputs_count.emplace_back(terms.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_result.data()));
    taskDataGlob->outputs_count.emplace_back(m_result.size());
  }

  sidorina_p_broadcast_mpi::Broadcast testMpiTaskParallel(taskDataGlob);
  testMpiTaskParallel.broadcast_fn = [](const boost::mpi::communicator& comm, int* values, int n, int root) {
    sidorina_p_broadcast_mpi::Broadcast::broadcast_m(comm, values, n, root);
  };
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    result.resize(array.size(), 0);

    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataRef->inputs_count.emplace_back(array.size());
    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskDataRef->inputs_count.emplace_back(terms.size());
    taskDataRef->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataRef->outputs_count.emplace_back(result.size());
  }
  sidorina_p_broadcast_mpi::Broadcast testMpiTaskSequential(taskDataRef);
  testMpiTaskSequential.broadcast_fn = [](const boost::mpi::communicator& comm, int* values, int n, int root) {
    boost::mpi::broadcast(comm, values, n, root);
  };
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(m_result, result);
  }
}

std::vector<int> randomVector(size_t size) {
  std::vector<int> v(size);
  std::random_device r;
  generate(v.begin(), v.end(), [&] { return r(); });
  return v;
}

TEST(sidorina_p_broadcast_mpi, Test_random) {
  boost::mpi::communicator world;

  std::vector<int> array;
  std::vector<int> terms;
  std::vector<int> m_result;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    array = std::vector<int>(0);
    array.resize(100, 0);
    for (unsigned long i = 0; i < array.size(); i++) {
      array[i] = round(rand() % 100);
    }
    terms = std::vector<int>(0);
    terms.resize(100, 0);
    for (unsigned long j = 0; j < terms.size(); j++) {
      terms[j] = round(rand() % 100);
    }

    m_result.resize(array.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataGlob->inputs_count.emplace_back(array.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskDataGlob->inputs_count.emplace_back(terms.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_result.data()));
    taskDataGlob->outputs_count.emplace_back(m_result.size());
  }

  sidorina_p_broadcast_mpi::Broadcast testMpiTaskParallel(taskDataGlob);
  testMpiTaskParallel.broadcast_fn = [](const boost::mpi::communicator& comm, int* values, int n, int root) {
    sidorina_p_broadcast_mpi::Broadcast::broadcast_m(comm, values, n, root);
  };
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    result.resize(array.size(), 0);

    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataRef->inputs_count.emplace_back(array.size());
    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskDataRef->inputs_count.emplace_back(terms.size());
    taskDataRef->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataRef->outputs_count.emplace_back(result.size());
  }
  sidorina_p_broadcast_mpi::Broadcast testMpiTaskSequential(taskDataRef);
  testMpiTaskSequential.broadcast_fn = [](const boost::mpi::communicator& comm, int* values, int n, int root) {
    boost::mpi::broadcast(comm, values, n, root);
  };
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(m_result, result);
  }
}

TEST(sidorina_p_broadcast_mpi, Test_validation_array_1) {
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
TEST(sidorina_p_broadcast_mpi, Test_validation_terms_1) {
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

TEST(sidorina_p_broadcast_mpi, Test_validation_1) {
  boost::mpi::communicator world;

  std::vector<int> array;
  std::vector<int> terms;
  std::vector<int> m_result;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    array = std::vector<int>();
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