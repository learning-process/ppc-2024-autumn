#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/shpynov_n_mismatched_characters_amount/include/mismatched_numbers_mpi.hpp"

TEST(Parallel_Operations_MPI, both_strings_empty) {
  // Create data
  boost::mpi::communicator world;

  std::string str1, str2 = "";
  std::vector<std::string> v1{str1, str2};

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();  // Create TaskData
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
    taskDataPar->inputs_count.emplace_back(v1.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  // Create Task
  shpynov_n_mismatched_numbers_mpi::TestMPITaskParallel testMPITaskPar(taskDataPar);
  ASSERT_EQ(testMPITaskPar.validation(), true);
  testMPITaskPar.pre_processing();
  testMPITaskPar.run();
  testMPITaskPar.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], 0);
  }
}

TEST(Parallel_Operations_MPI, only_one_string_is_empty) {
  boost::mpi::communicator world;

  std::string str1 = "abcd", str2 = "";
  std::vector<std::string> v1{str1, str2};

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
    taskDataPar->inputs_count.emplace_back(v1.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  shpynov_n_mismatched_numbers_mpi::TestMPITaskParallel testMPITaskPar(taskDataPar);
  ASSERT_EQ(testMPITaskPar.validation(), true);
  testMPITaskPar.pre_processing();
  testMPITaskPar.run();
  testMPITaskPar.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], (int)str1.length());
  }
}

TEST(Parallel_Operations_MPI, same_strings) {
  boost::mpi::communicator world;

  std::string str1 = "abcd", str2 = "abcd";
  std::vector<std::string> v1{str1, str2};

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
    taskDataPar->inputs_count.emplace_back(v1.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  shpynov_n_mismatched_numbers_mpi::TestMPITaskParallel testMPITaskPar(taskDataPar);
  ASSERT_EQ(testMPITaskPar.validation(), true);
  testMPITaskPar.pre_processing();
  testMPITaskPar.run();
  testMPITaskPar.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], 0);
  }
}

TEST(Parallel_Operations_MPI, correct_maths_4) {
  boost::mpi::communicator world;

  std::string str1 = "abcdffr";
  std::string str2 = "abcevvy";
  std::vector<std::string> v1{str1, str2};

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
    taskDataPar->inputs_count.emplace_back(v1.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  shpynov_n_mismatched_numbers_mpi::TestMPITaskParallel testMPITaskPar(taskDataPar);
  ASSERT_EQ(testMPITaskPar.validation(), true);
  testMPITaskPar.pre_processing();
  testMPITaskPar.run();
  testMPITaskPar.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], 4);
  }
}

TEST(Parallel_Operations_MPI, correct_maths_4_diff_lengths) {
  boost::mpi::communicator world;

  std::string str1 = "abcdf";
  std::string str2 = "abcevvy";
  std::vector<std::string> v1{str1, str2};

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
    taskDataPar->inputs_count.emplace_back(v1.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  shpynov_n_mismatched_numbers_mpi::TestMPITaskParallel testMPITaskPar(taskDataPar);
  ASSERT_EQ(testMPITaskPar.validation(), true);
  testMPITaskPar.pre_processing();
  testMPITaskPar.run();
  testMPITaskPar.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], 4);
  }
}

TEST(Parallel_Operations_MPI, correct_maths_8) {
  boost::mpi::communicator world;

  std::string str1 = "aebdfcuvgdhe";
  std::string str2 = "acbdbcedhugn";
  std::vector<std::string> v1{str1, str2};

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
    taskDataPar->inputs_count.emplace_back(v1.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  shpynov_n_mismatched_numbers_mpi::TestMPITaskParallel testMPITaskPar(taskDataPar);
  ASSERT_EQ(testMPITaskPar.validation(), true);
  testMPITaskPar.pre_processing();
  testMPITaskPar.run();
  testMPITaskPar.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], 8);
  }
}

TEST(Parallel_Operations_MPI, correct_maths_8_diff_lengths) {
  boost::mpi::communicator world;

  std::string str1 = "aebdfcuvgdhe";
  std::string str2 = "acbdbcedh";
  std::vector<std::string> v1{str1, str2};

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
    taskDataPar->inputs_count.emplace_back(v1.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  shpynov_n_mismatched_numbers_mpi::TestMPITaskParallel testMPITaskPar(taskDataPar);
  ASSERT_EQ(testMPITaskPar.validation(), true);
  testMPITaskPar.pre_processing();
  testMPITaskPar.run();
  testMPITaskPar.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], 8);
  }
}

TEST(Parallel_Operations_MPI, correct_maths_12) {
  boost::mpi::communicator world;

  std::string str1 = "aebdfc@egddddier/v";
  std::string str2 = "aedvfc@vnjgbj!eaew";
  std::vector<std::string> v1{str1, str2};

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
    taskDataPar->inputs_count.emplace_back(v1.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  shpynov_n_mismatched_numbers_mpi::TestMPITaskParallel testMPITaskPar(taskDataPar);
  ASSERT_EQ(testMPITaskPar.validation(), true);
  testMPITaskPar.pre_processing();
  testMPITaskPar.run();
  testMPITaskPar.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], 12);
  }
}

TEST(Parallel_Operations_MPI, correct_maths_12_diff_lengths) {
  boost::mpi::communicator world;

  std::string str1 = "aebdfc@egddddier/v";
  std::string str2 = "aedvfc@vnjgbj!e";
  std::vector<std::string> v1{str1, str2};

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
    taskDataPar->inputs_count.emplace_back(v1.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  shpynov_n_mismatched_numbers_mpi::TestMPITaskParallel testMPITaskPar(taskDataPar);
  ASSERT_EQ(testMPITaskPar.validation(), true);
  testMPITaskPar.pre_processing();
  testMPITaskPar.run();
  testMPITaskPar.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], 12);
  }
}

TEST(Parallel_Operations_MPI, parrallel_is_equal_to_sequential) {
  boost::mpi::communicator world;

  std::string str1 = "aebdfc@egddddier/v";
  std::string str2 = "aedvfc@vnjgbj!e";
  std::vector<std::string> v1{str1, str2};

  std::vector<int> out(1, 0), out_seq(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
    taskDataPar->inputs_count.emplace_back(v1.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  shpynov_n_mismatched_numbers_mpi::TestMPITaskParallel testMPITaskPar(taskDataPar);
  ASSERT_EQ(testMPITaskPar.validation(), true);
  testMPITaskPar.pre_processing();
  testMPITaskPar.run();
  testMPITaskPar.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
    taskDataSeq->inputs_count.emplace_back(v1.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out.size());

    shpynov_n_mismatched_numbers_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);

    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
  }

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], out_seq[0]);
  }
}
