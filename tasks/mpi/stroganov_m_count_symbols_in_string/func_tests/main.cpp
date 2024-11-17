// Copyright 2024 Stroganov Mikhail
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/stroganov_m_count_symbols_in_string/include/ops_mpi.hpp"
#include "mpi/stroganov_m_count_symbols_in_string/src/ops_mpi.cpp"

TEST(stroganov_m_count_symbols_in_string_mpi, EmptyString) {
  boost::mpi::communicator world;
  std::string global_str;

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataMpi->inputs_count.emplace_back(global_str.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }
  // Create Task
  stroganov_m_count_symbols_in_string_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(0, global_out[0]);
  }
}

TEST(stroganov_m_count_symbols_in_string_mpi, StringWithoutLetter) {
  boost::mpi::communicator world;
  std::string global_str = "1234";

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataMpi->inputs_count.emplace_back(global_str.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  stroganov_m_count_symbols_in_string_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    stroganov_m_count_symbols_in_string_mpi::TestMPITaskSequential TestMPITaskSequential(taskDataSeq);
    ASSERT_EQ(TestMPITaskSequential.validation(), true);
    TestMPITaskSequential.pre_processing();
    TestMPITaskSequential.run();
    TestMPITaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}

TEST(stroganov_m_count_symbols_in_string_mpi, StringWithOnlyLetter) {
  boost::mpi::communicator world;
  std::string global_str = "qwer";

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataMpi->inputs_count.emplace_back(global_str.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  stroganov_m_count_symbols_in_string_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    stroganov_m_count_symbols_in_string_mpi::TestMPITaskSequential TestMPITaskSequential(taskDataSeq);
    ASSERT_EQ(TestMPITaskSequential.validation(), true);
    TestMPITaskSequential.pre_processing();
    TestMPITaskSequential.run();
    TestMPITaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}

TEST(stroganov_m_count_symbols_in_string_mpi, RandomString) {
  boost::mpi::communicator world;
  std::string global_str = getRandomStringForCountOfSymbols();
  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataMpi->inputs_count.emplace_back(global_str.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  stroganov_m_count_symbols_in_string_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    stroganov_m_count_symbols_in_string_mpi::TestMPITaskSequential TestMPITaskSequential(taskDataSeq);
    ASSERT_EQ(TestMPITaskSequential.validation(), true);
    TestMPITaskSequential.pre_processing();
    TestMPITaskSequential.run();
    TestMPITaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}

TEST(stroganov_m_count_symbols_in_string_mpi, MixedSymbolsString) {
  boost::mpi::communicator world;
  std::string global_str = "qwer1234";

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataMpi->inputs_count.emplace_back(global_str.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  stroganov_m_count_symbols_in_string_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 4);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    stroganov_m_count_symbols_in_string_mpi::TestMPITaskSequential TestMPITaskSequential(taskDataSeq);
    ASSERT_EQ(TestMPITaskSequential.validation(), true);
    TestMPITaskSequential.pre_processing();
    TestMPITaskSequential.run();
    TestMPITaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}

TEST(stroganov_m_count_symbols_in_string_mpi, LongString) {
  boost::mpi::communicator world;
  std::string global_str(10000, 'a');

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataMpi->inputs_count.emplace_back(global_str.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  stroganov_m_count_symbols_in_string_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 10000);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    stroganov_m_count_symbols_in_string_mpi::TestMPITaskSequential TestMPITaskSequential(taskDataSeq);
    ASSERT_EQ(TestMPITaskSequential.validation(), true);
    TestMPITaskSequential.pre_processing();
    TestMPITaskSequential.run();
    TestMPITaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}

TEST(stroganov_m_count_symbols_in_string_mpi, OneLetterSingleString) {
  boost::mpi::communicator world;
  std::string global_str = "q";

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataMpi->inputs_count.emplace_back(global_str.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  stroganov_m_count_symbols_in_string_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    stroganov_m_count_symbols_in_string_mpi::TestMPITaskSequential TestMPITaskSequential(taskDataSeq);
    ASSERT_EQ(TestMPITaskSequential.validation(), true);
    TestMPITaskSequential.pre_processing();
    TestMPITaskSequential.run();
    TestMPITaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}

TEST(stroganov_m_count_symbols_in_string_mpi, TwoLetterSingleString) {
  boost::mpi::communicator world;
  std::string global_str = "qq";

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataMpi->inputs_count.emplace_back(global_str.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  stroganov_m_count_symbols_in_string_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    stroganov_m_count_symbols_in_string_mpi::TestMPITaskSequential TestMPITaskSequential(taskDataSeq);
    ASSERT_EQ(TestMPITaskSequential.validation(), true);
    TestMPITaskSequential.pre_processing();
    TestMPITaskSequential.run();
    TestMPITaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}

TEST(stroganov_m_count_symbols_in_string_mpi, ThreeLetterSingleString) {
  boost::mpi::communicator world;
  std::string global_str = "qqq";

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataMpi->inputs_count.emplace_back(global_str.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  stroganov_m_count_symbols_in_string_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    stroganov_m_count_symbols_in_string_mpi::TestMPITaskSequential TestMPITaskSequential(taskDataSeq);
    ASSERT_EQ(TestMPITaskSequential.validation(), true);
    TestMPITaskSequential.pre_processing();
    TestMPITaskSequential.run();
    TestMPITaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}

TEST(stroganov_m_count_symbols_in_string_mpi, FourLetterSingleString) {
  boost::mpi::communicator world;
  std::string global_str = "qqqq";

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataMpi->inputs_count.emplace_back(global_str.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  stroganov_m_count_symbols_in_string_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    stroganov_m_count_symbols_in_string_mpi::TestMPITaskSequential TestMPITaskSequential(taskDataSeq);
    ASSERT_EQ(TestMPITaskSequential.validation(), true);
    TestMPITaskSequential.pre_processing();
    TestMPITaskSequential.run();
    TestMPITaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}

TEST(stroganov_m_count_symbols_in_string_mpi, SingleStringWithoutLetter) {
  boost::mpi::communicator world;
  std::string global_str = "1";

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataMpi->inputs_count.emplace_back(global_str.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  stroganov_m_count_symbols_in_string_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    stroganov_m_count_symbols_in_string_mpi::TestMPITaskSequential TestMPITaskSequential(taskDataSeq);
    ASSERT_EQ(TestMPITaskSequential.validation(), true);
    TestMPITaskSequential.pre_processing();
    TestMPITaskSequential.run();
    TestMPITaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}
