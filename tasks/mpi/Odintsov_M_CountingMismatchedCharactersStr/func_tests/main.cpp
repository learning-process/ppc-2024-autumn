
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/Odintsov_M_CountingMismatchedCharactersStr/include/ops_mpi.hpp"

std::string get_random_str(size_t sz) {
  const char characters[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrswxyz0123456789";
  std::string str;

  std::srand(std::time(nullptr));

  for (size_t i = 0; i < sz; ++i) {
    // Генерируем случайный индекс
    int index = std::rand() % (sizeof(characters) - 1);
    str = characters[index];
  }

  return str;
}
TEST(Parallel_MPI_count, sz_0) {
  // Create data//
  boost::mpi::communicator com;
  char str1[] = "";
  char str2[] = "";
  std::vector<char*> in{str1, str2};
  std::vector<int> out(1, 1);
  std::vector<int> out_s(1, 1);
  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  Odintsov_M_CountingMismatchedCharactersStr_mpi::CountingCharacterMPIParallel testClassPar(taskDataPar);
  ASSERT_EQ(testClassPar.validation(), true);
  testClassPar.pre_processing();
  testClassPar.run();
  testClassPar.post_processing();

  if (com.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_s.data()));
    taskDataSeq->outputs_count.emplace_back(out_s.size());
    Odintsov_M_CountingMismatchedCharactersStr_mpi::CountingCharacterMPISequential testClassSeq(taskDataSeq);
    ASSERT_EQ(testClassSeq.validation(), true);
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    ASSERT_EQ(out[0], out_s[0]);
  }
}

TEST(Parallel_MPI_count, sz_1) {
  // Create data//
  boost::mpi::communicator com;
  std::string s1 = get_random_str(1);
  std::string s2 = get_random_str(1);
  std::vector<char*> in{s1.data(), s2.data()};
  std::vector<int> out(1, 1);
  std::vector<int> out_s(1, 1);
  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  Odintsov_M_CountingMismatchedCharactersStr_mpi::CountingCharacterMPIParallel testClassPar(taskDataPar);
  ASSERT_EQ(testClassPar.validation(), true);
  testClassPar.pre_processing();
  testClassPar.run();
  testClassPar.post_processing();

  if (com.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_s.data()));
    taskDataSeq->outputs_count.emplace_back(out_s.size());
    Odintsov_M_CountingMismatchedCharactersStr_mpi::CountingCharacterMPISequential testClassSeq(taskDataSeq);
    ASSERT_EQ(testClassSeq.validation(), true);
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    ASSERT_EQ(out[0], out_s[0]);
  }
}
TEST(Parallel_MPI_count, sz_36) {
  // Create data//
  boost::mpi::communicator com;
  std::string s1 = get_random_str(36);
  std::string s2 = get_random_str(36);
  std::vector<char*> in{s1.data(), s2.data()};
  std::vector<int> out(1, 1);
  std::vector<int> out_s(1, 1);
  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  Odintsov_M_CountingMismatchedCharactersStr_mpi::CountingCharacterMPIParallel testClassPar(taskDataPar);
  ASSERT_EQ(testClassPar.validation(), true);
  testClassPar.pre_processing();
  testClassPar.run();
  testClassPar.post_processing();

  if (com.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_s.data()));
    taskDataSeq->outputs_count.emplace_back(out_s.size());
    Odintsov_M_CountingMismatchedCharactersStr_mpi::CountingCharacterMPISequential testClassSeq(taskDataSeq);
    ASSERT_EQ(testClassSeq.validation(), true);
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    ASSERT_EQ(out[0], out_s[0]);
  }
}

TEST(Parallel_MPI_count, sz_24) {
  // Create data
  boost::mpi::communicator com;
  std::vector<int> out_s(1, 1);
  std::string s1 = get_random_str(24);
  std::string s2 = get_random_str(24);

  std::vector<char*> in{s1.data(), s2.data()};
  std::vector<int> out(1, 1);

  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  Odintsov_M_CountingMismatchedCharactersStr_mpi::CountingCharacterMPIParallel testClassPar(taskDataPar);
  ASSERT_EQ(testClassPar.validation(), true);
  testClassPar.pre_processing();
  testClassPar.run();
  testClassPar.post_processing();

  if (com.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_s.data()));
    taskDataSeq->outputs_count.emplace_back(out_s.size());
    Odintsov_M_CountingMismatchedCharactersStr_mpi::CountingCharacterMPISequential testClassSeq(taskDataSeq);
    ASSERT_EQ(testClassSeq.validation(), true);
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    ASSERT_EQ(out[0], out_s[0]);
  }
}
TEST(Parallel_MPI_count, df_sz_15) {
  // Create data//
  boost::mpi::communicator com;
  std::string s1 = get_random_str(12);
  std::string s2 = get_random_str(12);

  std::vector<char*> in{s1.data(), s2.data()};
  std::vector<int> out(1, 1);
  std::vector<int> out_s(1, 1);
  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  Odintsov_M_CountingMismatchedCharactersStr_mpi::CountingCharacterMPIParallel testClassPar(taskDataPar);
  ASSERT_EQ(testClassPar.validation(), true);
  testClassPar.pre_processing();
  testClassPar.run();
  testClassPar.post_processing();

  if (com.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_s.data()));
    taskDataSeq->outputs_count.emplace_back(out_s.size());
    Odintsov_M_CountingMismatchedCharactersStr_mpi::CountingCharacterMPISequential testClassSeq(taskDataSeq);
    ASSERT_EQ(testClassSeq.validation(), true);
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    ASSERT_EQ(out[0], out_s[0]);
  }
}
TEST(Parallel_MPI_count, df_sz_25) {
  // Create data//
  boost::mpi::communicator com;
  std::string s1 = get_random_str(13);
  std::string s2 = get_random_str(12);

  std::vector<char*> in{s1.data(), s2.data()};
  std::vector<int> out(1, 1);
  std::vector<int> out_s(1, 1);
  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  Odintsov_M_CountingMismatchedCharactersStr_mpi::CountingCharacterMPIParallel testClassPar(taskDataPar);
  ASSERT_EQ(testClassPar.validation(), true);
  testClassPar.pre_processing();
  testClassPar.run();
  testClassPar.post_processing();

  if (com.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_s.data()));
    taskDataSeq->outputs_count.emplace_back(out_s.size());
    Odintsov_M_CountingMismatchedCharactersStr_mpi::CountingCharacterMPISequential testClassSeq(taskDataSeq);
    ASSERT_EQ(testClassSeq.validation(), true);
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    ASSERT_EQ(out[0], out_s[0]);
  }
}