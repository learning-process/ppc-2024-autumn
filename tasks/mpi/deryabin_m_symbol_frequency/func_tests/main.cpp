#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/deryabin_m_symbol_frequency/include/ops_mpi.hpp"

TEST(deryabin_m_symbol_frequency_mpi, test_shuffle) {
  boost::mpi::communicator world;
  std::vector<char> global_str;
  std::vector<char> input_symbol(1, 'A');
  std::vector<int32_t> global_frequency(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_str = {'A', '1', 'A', '3', 'A', '5', 'A', '7', 'A', '9', 'A', 'B', 'A', 'D', 'A', 'F',
                  'A', 'H', 'A', 'J', 'A', 'L', 'A', 'N', 'A', 'P', 'A', 'R', 'A', 'T', 'A', 'V',
                  'A', 'X', 'A', 'Z', 'A', 'b', 'A', 'd', 'A', 'f', 'A', 'h', 'A', 'j', 'A', 'l',
                  'A', 'n', 'A', 'p', 'A', 'r', 'A', 't', 'A', 'v', 'A', 'x', 'A', 'z'};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(global_str.begin(), global_str.end(), gen);
    uniform_int_distribution<> distrib(1, 62);
    std::vector<char>::const_iterator first = global_str.begin() + distrib(gen);
    std::vector<char>::const_iterator last = global_str.end();
    global_str.erase(first, last);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_symbol.data()));
    taskDataPar->inputs_count.emplace_back(input_symbol.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_frequency.data()));
    taskDataPar->outputs_count.emplace_back(global_frequency.size());
  }

  deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_frequency(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_symbol.data()));
    taskDataSeq->inputs_count.emplace_back(input_symbol.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_frequency.data()));
    taskDataSeq->outputs_count.emplace_back(reference_frequency.size());

    // Create Task
    deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_frequency[0], global_frequency[0]);
  }
}

TEST(deryabin_m_symbol_frequency_mpi, test_empty) {
  boost::mpi::communicator world;
  std::vector<char> global_str;
  std::vector<char> input_symbol(1, 'A');
  std::vector<int32_t> global_frequency(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_symbol.data()));
    taskDataPar->inputs_count.emplace_back(input_symbol.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_frequency.data()));
    taskDataPar->outputs_count.emplace_back(global_frequency.size());
  }

  deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  ASSERT_EQ(0, global_frequency[0]);
}

TEST(deryabin_m_symbol_frequency_mpi, test_every_secondary) {
  boost::mpi::communicator world;
  std::vector<char> global_str;
  std::vector<char> input_symbol(1, 'A');
  std::vector<int32_t> global_frequency(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_str = {'1', '2', '3', '4', '5', '6', '7', '8', '9', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                  'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
                  'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
    std::random_device rd;
    std::mt19937 generator(rd());
    std::shuffle(global_str.begin(), global_str.end(), generator);
    for (size_t i = 0; i < global_str.size(); i++) {
      if (i % 2 == 0) {
        global_str[i] = input_symbol[0];
      }
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_symbol.data()));
    taskDataPar->inputs_count.emplace_back(input_symbol.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_frequency.data()));
    taskDataPar->outputs_count.emplace_back(global_frequency.size());
  }

  deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_frequency(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_symbol.data()));
    taskDataSeq->inputs_count.emplace_back(input_symbol.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_frequency.data()));
    taskDataSeq->outputs_count.emplace_back(reference_frequency.size());

    // Create Task
    deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_frequency[0], global_frequency[0]);
  }
}

TEST(deryabin_m_symbol_frequency_mpi, test_same_letters) {
  boost::mpi::communicator world;
  std::vector<char> global_str;
  std::vector<char> input_symbol(1, 'A');
  std::vector<int32_t> global_frequency(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_str = std::vector<char>(1000, 'A');
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_symbol.data()));
    taskDataPar->inputs_count.emplace_back(input_symbol.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_frequency.data()));
    taskDataPar->outputs_count.emplace_back(global_frequency.size());
  }

  deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_frequency(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_symbol.data()));
    taskDataSeq->inputs_count.emplace_back(input_symbol.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_frequency.data()));
    taskDataSeq->outputs_count.emplace_back(reference_frequency.size());

    // Create Task
    deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_frequency[0], global_frequency[0]);
  }
}

TEST(deryabin_m_symbol_frequency_mpi, test_random) {
  boost::mpi::communicator world;
  std::vector<char> global_str;
  std::vector<char> input_symbol(1, 'A');
  std::vector<int32_t> global_frequency(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_str = std::vector<char>(10000);
    std::default_random_engine generator(rand());
    std::uniform_int_distribution<> distribution(65, 90);
    std::generate(global_str.begin(), global_str.end(), [&] { return char(0) + distribution(generator); });
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_symbol.data()));
    taskDataPar->inputs_count.emplace_back(input_symbol.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_frequency.data()));
    taskDataPar->outputs_count.emplace_back(global_frequency.size());
  }

  deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_frequency(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_symbol.data()));
    taskDataSeq->inputs_count.emplace_back(input_symbol.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_frequency.data()));
    taskDataSeq->outputs_count.emplace_back(reference_frequency.size());

    // Create Task
    deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_frequency[0], global_frequency[0]);
  }
}
