#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm/include/ops_mpi.hpp"

namespace alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm {
std::vector<double> generator(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());

  std::uniform_real_distribution<double> dis(-100.0, 100.0);

  std::vector<double> ans(sz);
  for (int i = 0; i < sz; ++i) {
    ans[i] = dis(gen);
  }

  return ans;
}
}  // namespace alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm

TEST(alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm, EmptyInput_ReturnsFalse1) {
  boost::mpi::communicator world;
  std::vector<double> A;
  std::vector<double> B;
  int N = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<int> reference_ans(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs_count.emplace_back(N);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_ans.data()));
    taskDataSeq->outputs_count.emplace_back(reference_ans.size());

    alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
        dense_matrix_multiplication_block_scheme_fox_algorithm_seq
            dense_matrix_multiplication_block_scheme_fox_algorithm_seq(taskDataSeq);
    ASSERT_EQ(dense_matrix_multiplication_block_scheme_fox_algorithm_seq.validation(), false);
  }
}

TEST(alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm, EmptyInput_ReturnsFalse2) {
  boost::mpi::communicator world;
  std::vector<double> A;
  int N = 1;
  std::vector<double> B;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<int> reference_ans(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs_count.emplace_back(N);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_ans.data()));
    taskDataSeq->outputs_count.emplace_back(reference_ans.size());

    alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
        dense_matrix_multiplication_block_scheme_fox_algorithm_seq
            dense_matrix_multiplication_block_scheme_fox_algorithm_seq(taskDataSeq);
    ASSERT_EQ(dense_matrix_multiplication_block_scheme_fox_algorithm_seq.validation(), false);
  }
}

TEST(alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm, InputSizeTwo_CorrectResult) {
  boost::mpi::communicator world;
  int x = static_cast<int>(std::sqrt(static_cast<double>(world.rank())));
  if (x * x != world.rank()) {
    GTEST_SKIP();
  }
  std::vector<double> A = alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::generator(4 * 4);
  int N = 4;
  std::vector<double> B = alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::generator(4 * 4);
  std::vector<double> ansPar(4 * 4);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // for (int i = 0; i < 4; ++i) {
    // std::cout << A[i] << " " << B[i] << std::endl;
    //}

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs_count.emplace_back(N);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ansPar.data()));
    taskDataPar->outputs_count.emplace_back(ansPar.size());
  }

  alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
      dense_matrix_multiplication_block_scheme_fox_algorithm_mpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> ansSeq(4 * 4);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs_count.emplace_back(N);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ansSeq.data()));
    taskDataSeq->outputs_count.emplace_back(ansSeq.size());

    alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
        dense_matrix_multiplication_block_scheme_fox_algorithm_seq
            dense_matrix_multiplication_block_scheme_fox_algorithm_seq(taskDataSeq);
    ASSERT_EQ(dense_matrix_multiplication_block_scheme_fox_algorithm_seq.validation(), true);
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq.pre_processing();
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq.run();
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq.post_processing();

    // for (int i = 0; i < 4; ++i) {
    // std::cout << ansPar[i] << " " << ansSeq[i] << std::endl;
    //}

    for (int i = 0; i < 4 * 4; ++i) {
      ASSERT_NEAR(ansPar[i], ansSeq[i], 1e-5);
    }
  }
}

TEST(alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm, LargeRandomInput_CorrectResult) {
  boost::mpi::communicator world;
  int x = static_cast<int>(std::sqrt(static_cast<double>(world.rank())));
  if (x * x != world.rank()) {
    GTEST_SKIP();
  }
  int N = 25;
  std::vector<double> A = alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::generator(N * N);
  std::vector<double> B = alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::generator(N * N);
  std::vector<double> ansPar(N * N);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // for (int i = 0; i < N * N; ++i) {
    // std::cout << A[i] << " " << B[i] << std::endl;
    //}

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs_count.emplace_back(N);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ansPar.data()));
    taskDataPar->outputs_count.emplace_back(ansPar.size());
  }

  alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
      dense_matrix_multiplication_block_scheme_fox_algorithm_mpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> ansSeq(N * N);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs_count.emplace_back(N);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ansSeq.data()));
    taskDataSeq->outputs_count.emplace_back(ansSeq.size());

    alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
        dense_matrix_multiplication_block_scheme_fox_algorithm_seq
            dense_matrix_multiplication_block_scheme_fox_algorithm_seq(taskDataSeq);
    ASSERT_EQ(dense_matrix_multiplication_block_scheme_fox_algorithm_seq.validation(), true);
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq.pre_processing();
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq.run();
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq.post_processing();

    // for (int i = 0; i < N * N; ++i) {
    // std::cout << ansPar[i] << " " << ansSeq[i] << std::endl;
    //}

    for (int i = 0; i < N * N; ++i) {
      ASSERT_NEAR(ansPar[i], ansSeq[i], 1e-5);
    }
  }
}

TEST(alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm, MediumRandomInput_CorrectResult) {
  boost::mpi::communicator world;
  int x = static_cast<int>(std::sqrt(static_cast<double>(world.rank())));
  if (x * x != world.rank()) {
    GTEST_SKIP();
  }
  int N = 13;
  std::vector<double> A = alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::generator(N * N);
  std::vector<double> B = alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::generator(N * N);
  std::vector<double> ansPar(N * N);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // for (int i = 0; i < N * N; ++i) {
    // std::cout << A[i] << " " << B[i] << std::endl;
    //}

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs_count.emplace_back(N);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ansPar.data()));
    taskDataPar->outputs_count.emplace_back(ansPar.size());
  }

  alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
      dense_matrix_multiplication_block_scheme_fox_algorithm_mpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> ansSeq(N * N);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs_count.emplace_back(N);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ansSeq.data()));
    taskDataSeq->outputs_count.emplace_back(ansSeq.size());

    alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
        dense_matrix_multiplication_block_scheme_fox_algorithm_seq
            dense_matrix_multiplication_block_scheme_fox_algorithm_seq(taskDataSeq);
    ASSERT_EQ(dense_matrix_multiplication_block_scheme_fox_algorithm_seq.validation(), true);
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq.pre_processing();
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq.run();
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq.post_processing();

    // for (int i = 0; i < N * N; ++i) {
    // std::cout << ansPar[i] << " " << ansSeq[i] << std::endl;
    //}

    for (int i = 0; i < N * N; ++i) {
      ASSERT_NEAR(ansPar[i], ansSeq[i], 1e-5);
    }
  }
}

TEST(alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm, AllEqualElements_CorrectResult) {
  boost::mpi::communicator world;
  int x = static_cast<int>(std::sqrt(static_cast<double>(world.rank())));
  if (x * x != world.rank()) {
    GTEST_SKIP();
  }
  int N = 12;
  std::vector<double> A = alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::generator(N * N);
  std::vector<double> B = alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::generator(N * N);
  std::vector<double> ansPar(N * N);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // for (int i = 0; i < N * N; ++i) {
    // std::cout << A[i] << " " << B[i] << std::endl;
    //}

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs_count.emplace_back(N);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ansPar.data()));
    taskDataPar->outputs_count.emplace_back(ansPar.size());
  }

  alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
      dense_matrix_multiplication_block_scheme_fox_algorithm_mpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> ansSeq(N * N);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs_count.emplace_back(N);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ansSeq.data()));
    taskDataSeq->outputs_count.emplace_back(ansSeq.size());

    alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
        dense_matrix_multiplication_block_scheme_fox_algorithm_seq
            dense_matrix_multiplication_block_scheme_fox_algorithm_seq(taskDataSeq);
    ASSERT_EQ(dense_matrix_multiplication_block_scheme_fox_algorithm_seq.validation(), true);
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq.pre_processing();
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq.run();
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq.post_processing();

    // for (int i = 0; i < N * N; ++i) {
    // std::cout << ansPar[i] << " " << ansSeq[i] << std::endl;
    //}

    for (int i = 0; i < N * N; ++i) {
      ASSERT_NEAR(ansPar[i], ansSeq[i], 1e-5);
    }
  }
}

TEST(alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm, AlternatingElements_CorrectResult) {
  boost::mpi::communicator world;
  int x = static_cast<int>(std::sqrt(static_cast<double>(world.rank())));
  if (x * x != world.rank()) {
    GTEST_SKIP();
  }
  int N = 4;
  std::vector<double> A = alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::generator(N * N);
  std::vector<double> B = alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::generator(N * N);
  std::vector<double> ansPar(N * N);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // for (int i = 0; i < N * N; ++i) {
    // std::cout << A[i] << " " << B[i] << std::endl;
    //}

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs_count.emplace_back(N);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ansPar.data()));
    taskDataPar->outputs_count.emplace_back(ansPar.size());
  }

  alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
      dense_matrix_multiplication_block_scheme_fox_algorithm_mpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> ansSeq(N * N);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs_count.emplace_back(N);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ansSeq.data()));
    taskDataSeq->outputs_count.emplace_back(ansSeq.size());

    alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
        dense_matrix_multiplication_block_scheme_fox_algorithm_seq
            dense_matrix_multiplication_block_scheme_fox_algorithm_seq(taskDataSeq);
    ASSERT_EQ(dense_matrix_multiplication_block_scheme_fox_algorithm_seq.validation(), true);
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq.pre_processing();
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq.run();
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq.post_processing();

    // for (int i = 0; i < N * N; ++i) {
    // std::cout << ansPar[i] << " " << ansSeq[i] << std::endl;
    //}

    for (int i = 0; i < N * N; ++i) {
      ASSERT_NEAR(ansPar[i], ansSeq[i], 1e-5);
    }
  }
}

TEST(alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm, ConstantDifferenceSequence_CorrectResult) {
  boost::mpi::communicator world;
  int x = static_cast<int>(std::sqrt(static_cast<double>(world.rank())));
  if (x * x != world.rank()) {
    GTEST_SKIP();
  }
  int N = 8;
  std::vector<double> A = alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::generator(N * N);
  std::vector<double> B = alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::generator(N * N);
  std::vector<double> ansPar(N * N);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // for (int i = 0; i < N * N; ++i) {
    // std::cout << A[i] << " " << B[i] << std::endl;
    //}

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs_count.emplace_back(N);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ansPar.data()));
    taskDataPar->outputs_count.emplace_back(ansPar.size());
  }

  alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
      dense_matrix_multiplication_block_scheme_fox_algorithm_mpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> ansSeq(N * N);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs_count.emplace_back(N);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ansSeq.data()));
    taskDataSeq->outputs_count.emplace_back(ansSeq.size());

    alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
        dense_matrix_multiplication_block_scheme_fox_algorithm_seq
            dense_matrix_multiplication_block_scheme_fox_algorithm_seq(taskDataSeq);
    ASSERT_EQ(dense_matrix_multiplication_block_scheme_fox_algorithm_seq.validation(), true);
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq.pre_processing();
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq.run();
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq.post_processing();

    // for (int i = 0; i < N * N; ++i) {
    // std::cout << ansPar[i] << " " << ansSeq[i] << std::endl;
    //}

    for (int i = 0; i < N * N; ++i) {
      ASSERT_NEAR(ansPar[i], ansSeq[i], 1e-5);
    }
  }
}
