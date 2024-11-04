// Copyright 2024 Anikin Maksim
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>
#include <random>

#include "mpi/anikin_m_summ_of_different_symbols/include/ops_mpi.hpp"

std::string getRandomString(int size) {
  const std::string characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<> distribution(0, characters.size() - 1);

  std::string randomString;
  for (size_t i = 0; i < size; ++i) {
    randomString = characters[distribution(generator)];
  }
  return randomString;
}

TEST(anikin_m_Parallel_SummDifSym_count, size_0) {
  boost::mpi::communicator com;
  char str1[] = "";
  char str2[] = "";
  std::vector<char*> in{str1, str2};
  std::vector<int> out(1, 1);
  std::vector<int> out_s(1, 1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  anikin_m_summ_of_different_symbols_mpi::SumDifSymMPIParallel testClassPar(taskDataPar);
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
    anikin_m_summ_of_different_symbols_mpi::SumDifSymMPISequential testClassSeq(taskDataSeq);
    ASSERT_EQ(testClassSeq.validation(), true);
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    ASSERT_EQ(out[0], out_s[0]);
  }
}

TEST(anikin_m_Parallel_SummDifSym_count, size_25) {
  boost::mpi::communicator com;
  std::string s1 = getRandomString(25);
  std::string s2 = getRandomString(25);
  std::vector<char*> in{s1.data(), s2.data()};
  std::vector<int> out(1, 1);
  std::vector<int> out_s(1, 1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  anikin_m_summ_of_different_symbols_mpi::SumDifSymMPIParallel testClassPar(taskDataPar);
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
    anikin_m_summ_of_different_symbols_mpi::SumDifSymMPISequential testClassSeq(taskDataSeq);
    ASSERT_EQ(testClassSeq.validation(), true);
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    ASSERT_EQ(out[0], out_s[0]);
  }
}

TEST(anikin_m_Parallel_SummDifSym_count, first_larger) {
  boost::mpi::communicator com;
  std::string s1 = getRandomString(25);
  std::string s2 = getRandomString(19);
  std::vector<char*> in{s1.data(), s2.data()};
  std::vector<int> out(1, 1);
  std::vector<int> out_s(1, 1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  anikin_m_summ_of_different_symbols_mpi::SumDifSymMPIParallel testClassPar(taskDataPar);
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
    anikin_m_summ_of_different_symbols_mpi::SumDifSymMPISequential testClassSeq(taskDataSeq);
    ASSERT_EQ(testClassSeq.validation(), true);
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    ASSERT_EQ(out[0], out_s[0]);
  }
}

TEST(anikin_m_Parallel_SummDifSym_count, second_larger) {
  boost::mpi::communicator com;
  std::string s1 = getRandomString(20);
  std::string s2 = getRandomString(25);
  std::vector<char*> in{s1.data(), s2.data()};
  std::vector<int> out(1, 1);
  std::vector<int> out_s(1, 1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  anikin_m_summ_of_different_symbols_mpi::SumDifSymMPIParallel testClassPar(taskDataPar);
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
    anikin_m_summ_of_different_symbols_mpi::SumDifSymMPISequential testClassSeq(taskDataSeq);
    ASSERT_EQ(testClassSeq.validation(), true);
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    ASSERT_EQ(out[0], out_s[0]);
  }
}