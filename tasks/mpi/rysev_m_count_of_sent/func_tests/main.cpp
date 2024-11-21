// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/rysev_m_count_of_sent/include/ops_mpi.hpp"

TEST(rysev_m_count_of_sent_mpi, simple_test) {
  boost::mpi::communicator world;

  // create data
  std::string str = "The cake is a lie.";
  std::vector<int> par_out(1, 0);

  // create taskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataPar->inputs_count.emplace_back(str.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_out.data()));
  taskDataPar->outputs_count.emplace_back(par_out.size());

  // create task
  rysev_m_count_of_sent_mpi::CountOfSentParallel counter(taskDataPar);
  ASSERT_EQ(counter.validation(), true);
  counter.pre_processing();
  counter.run();
  counter.post_processing();
  ASSERT_EQ(par_out[0], 1);

  // compare with seq version
  if (world.rank() == 0) {
    std::vector<int> seq_out(1, 0);

    // create taskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
    taskDataSeq->inputs_count.emplace_back(str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    // create task
    rysev_m_count_of_sent_mpi::CountOfSentSeq counter1(taskDataSeq);
    ASSERT_EQ(counter1.validation(), true);
    counter1.pre_processing();
    counter1.run();
    counter1.post_processing();

    //compare
    ASSERT_EQ(par_out[0], seq_out[0]);
  }
}

TEST(rysev_m_count_of_sent_mpi, emty_string) {
  boost::mpi::communicator world;

  // create data
  std::string str = "";
  std::vector<int> par_out(1, 0);

  // create taskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataPar->inputs_count.emplace_back(str.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_out.data()));
  taskDataPar->outputs_count.emplace_back(par_out.size());

  // create task
  rysev_m_count_of_sent_mpi::CountOfSentParallel counter(taskDataPar);
  ASSERT_EQ(counter.validation(), true);
  counter.pre_processing();
  counter.run();
  counter.post_processing();
  ASSERT_EQ(par_out[0], 0);

  // compare with seq version
  if (world.rank() == 0) {
    std::vector<int> seq_out(1, 0);

    // create taskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
    taskDataSeq->inputs_count.emplace_back(str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    // create task
    rysev_m_count_of_sent_mpi::CountOfSentSeq counter1(taskDataSeq);
    ASSERT_EQ(counter1.validation(), true);
    counter1.pre_processing();
    counter1.run();
    counter1.post_processing();

    // compare
    ASSERT_EQ(par_out[0], seq_out[0]);
  }
}

TEST(rysev_m_count_of_sent_mpi, text_without_end_symbol) {
  boost::mpi::communicator world;

  // create data
  std::string str = "Bring me a bucket, and I'll show you a bucket";
  std::vector<int> par_out(1, 0);

  // create taskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataPar->inputs_count.emplace_back(str.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_out.data()));
  taskDataPar->outputs_count.emplace_back(par_out.size());

  // create task
  rysev_m_count_of_sent_mpi::CountOfSentParallel counter(taskDataPar);
  ASSERT_EQ(counter.validation(), true);
  counter.pre_processing();
  counter.run();
  counter.post_processing();
  ASSERT_EQ(par_out[0], 1);

  // compare with seq version
  if (world.rank() == 0) {
    std::vector<int> seq_out(1, 0);

    // create taskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
    taskDataSeq->inputs_count.emplace_back(str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    // create task
    rysev_m_count_of_sent_mpi::CountOfSentSeq counter1(taskDataSeq);
    ASSERT_EQ(counter1.validation(), true);
    counter1.pre_processing();
    counter1.run();
    counter1.post_processing();

    // compare
    ASSERT_EQ(par_out[0], seq_out[0]);
  }
}

TEST(rysev_m_count_of_sent_mpi, text_with_double_or_more_end_symbols) {
  boost::mpi::communicator world;

  // create data
  std::string str = "Who will you choose: them or us?! Us or them?!!...";
  std::vector<int> par_out(1, 0);

  // create taskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataPar->inputs_count.emplace_back(str.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_out.data()));
  taskDataPar->outputs_count.emplace_back(par_out.size());

  // create task
  rysev_m_count_of_sent_mpi::CountOfSentParallel counter(taskDataPar);
  ASSERT_EQ(counter.validation(), true);
  counter.pre_processing();
  counter.run();
  counter.post_processing();
  ASSERT_EQ(par_out[0], 2);

  // compare with seq version
  if (world.rank() == 0) {
    std::vector<int> seq_out(1, 0);

    // create taskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
    taskDataSeq->inputs_count.emplace_back(str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    // create task
    rysev_m_count_of_sent_mpi::CountOfSentSeq counter1(taskDataSeq);
    ASSERT_EQ(counter1.validation(), true);
    counter1.pre_processing();
    counter1.run();
    counter1.post_processing();

    // compare
    ASSERT_EQ(par_out[0], seq_out[0]);
  }
}

TEST(rysev_m_count_of_sent_mpi, sample_test_number_two) {
  boost::mpi::communicator world;

  // create data
  std::string str = "We both said a lot of things that you'll regret. But I think we can put our differences behind us. For the sake of science. You're a monster.";
  std::vector<int> par_out(1, 0);

  // create taskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataPar->inputs_count.emplace_back(str.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_out.data()));
  taskDataPar->outputs_count.emplace_back(par_out.size());

  // create task
  rysev_m_count_of_sent_mpi::CountOfSentParallel counter(taskDataPar);
  ASSERT_EQ(counter.validation(), true);
  counter.pre_processing();
  counter.run();
  counter.post_processing();
  ASSERT_EQ(par_out[0], 4);

  // compare with seq version
  if (world.rank() == 0) {
    std::vector<int> seq_out(1, 0);

    // create taskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
    taskDataSeq->inputs_count.emplace_back(str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    // create task
    rysev_m_count_of_sent_mpi::CountOfSentSeq counter1(taskDataSeq);
    ASSERT_EQ(counter1.validation(), true);
    counter1.pre_processing();
    counter1.run();
    counter1.post_processing();

    // compare
    ASSERT_EQ(par_out[0], seq_out[0]);
  }
}
