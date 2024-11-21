#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <gtest/gtest.h>
#include "mpi/rysev_m_count_of_sent/include/ops_mpi.hpp"
#include <vector>

TEST(rysev_m_count_of_sent_mpi, simple_test) {
  boost::mpi::communicator world;

  std::string str = "The cake is a lie.";
  std::vector<int> par_out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataPar->inputs_count.emplace_back(str.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_out.data()));
  taskDataPar->outputs_count.emplace_back(par_out.size());

  rysev_m_count_of_sent_mpi::CountOfSentParallel counter(taskDataPar);
  ASSERT_EQ(counter.validation(), true);
  counter.pre_processing();
  counter.run();
  counter.post_processing();
  ASSERT_EQ(par_out[0], 1);

  if (world.rank() == 0) {
    std::vector<int> seq_out(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
    taskDataSeq->inputs_count.emplace_back(str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    rysev_m_count_of_sent_mpi::CountOfSentSeq counter1(taskDataSeq);
    ASSERT_EQ(counter1.validation(), true);
    counter1.pre_processing();
    counter1.run();
    counter1.post_processing();

    ASSERT_EQ(par_out[0], seq_out[0]);
  }
}

TEST(rysev_m_count_of_sent_mpi, emty_string) {
  boost::mpi::communicator world;

  std::string str = "";
  std::vector<int> par_out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataPar->inputs_count.emplace_back(str.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_out.data()));
  taskDataPar->outputs_count.emplace_back(par_out.size());

  rysev_m_count_of_sent_mpi::CountOfSentParallel counter(taskDataPar);
  ASSERT_EQ(counter.validation(), true);
  counter.pre_processing();
  counter.run();
  counter.post_processing();
  ASSERT_EQ(par_out[0], 0);

  if (world.rank() == 0) {
    std::vector<int> seq_out(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
    taskDataSeq->inputs_count.emplace_back(str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    rysev_m_count_of_sent_mpi::CountOfSentSeq counter1(taskDataSeq);
    ASSERT_EQ(counter1.validation(), true);
    counter1.pre_processing();
    counter1.run();
    counter1.post_processing();

    ASSERT_EQ(par_out[0], seq_out[0]);
  }
}

TEST(rysev_m_count_of_sent_mpi, text_without_end_symbol) {
  boost::mpi::communicator world;

  std::string str = "Bring me a bucket, and I'll show you a bucket";
  std::vector<int> par_out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataPar->inputs_count.emplace_back(str.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_out.data()));
  taskDataPar->outputs_count.emplace_back(par_out.size());

  rysev_m_count_of_sent_mpi::CountOfSentParallel counter(taskDataPar);
  ASSERT_EQ(counter.validation(), true);
  counter.pre_processing();
  counter.run();
  counter.post_processing();
  ASSERT_EQ(par_out[0], 1);

  if (world.rank() == 0) {
    std::vector<int> seq_out(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
    taskDataSeq->inputs_count.emplace_back(str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    rysev_m_count_of_sent_mpi::CountOfSentSeq counter1(taskDataSeq);
    ASSERT_EQ(counter1.validation(), true);
    counter1.pre_processing();
    counter1.run();
    counter1.post_processing();

    ASSERT_EQ(par_out[0], seq_out[0]);
  }
}

TEST(rysev_m_count_of_sent_mpi, text_with_double_or_more_end_symbols) {
  boost::mpi::communicator world;

  std::string str = "Who will you choose: them or us?! Us or them?!!...";
  std::vector<int> par_out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataPar->inputs_count.emplace_back(str.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_out.data()));
  taskDataPar->outputs_count.emplace_back(par_out.size());

  rysev_m_count_of_sent_mpi::CountOfSentParallel counter(taskDataPar);
  ASSERT_EQ(counter.validation(), true);
  counter.pre_processing();
  counter.run();
  counter.post_processing();
  ASSERT_EQ(par_out[0], 2);

  if (world.rank() == 0) {
    std::vector<int> seq_out(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
    taskDataSeq->inputs_count.emplace_back(str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    rysev_m_count_of_sent_mpi::CountOfSentSeq counter1(taskDataSeq);
    ASSERT_EQ(counter1.validation(), true);
    counter1.pre_processing();
    counter1.run();
    counter1.post_processing();

    ASSERT_EQ(par_out[0], seq_out[0]);
  }
}

TEST(rysev_m_count_of_sent_mpi, sample_test_number_two) {
  boost::mpi::communicator world;

  std::string str = "We both said a lot of things that you'll regret. But I think we can put our differences behind us. For the sake of science. You're a monster.";
  std::vector<int> par_out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataPar->inputs_count.emplace_back(str.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_out.data()));
  taskDataPar->outputs_count.emplace_back(par_out.size());

  rysev_m_count_of_sent_mpi::CountOfSentParallel counter(taskDataPar);
  ASSERT_EQ(counter.validation(), true);
  counter.pre_processing();
  counter.run();
  counter.post_processing();
  ASSERT_EQ(par_out[0], 4);

  if (world.rank() == 0) {
    std::vector<int> seq_out(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
    taskDataSeq->inputs_count.emplace_back(str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    rysev_m_count_of_sent_mpi::CountOfSentSeq counter1(taskDataSeq);
    ASSERT_EQ(counter1.validation(), true);
    counter1.pre_processing();
    counter1.run();
    counter1.post_processing();

    ASSERT_EQ(par_out[0], seq_out[0]);
  }
}
