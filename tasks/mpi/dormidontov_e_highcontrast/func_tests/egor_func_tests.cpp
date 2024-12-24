#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/dormidontov_e_highcontrast/include/egor_include.hpp"

TEST(dormidontov_e_highcontrast_mpi, Test_just_test_if_it_finally_works) {
  boost::mpi::communicator world;
  int rs = 7;
  int cs = 7;

  std::vector<int> matrix(cs * rs);
  matrix = dormidontov_e_highcontrast_mpi::generate_halftone_pic(cs, rs);
  std::vector<int> res_out_paral(cs * rs, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(rs * cs);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }
  dormidontov_e_highcontrast_mpi::ContrastP ContrastP(taskDataPar);

  ASSERT_EQ(ContrastP.validation(), true);

  ContrastP.pre_processing();

  ContrastP.run();

  ContrastP.post_processing();
  if (world.rank() == 0) {
    std::vector<int> res_out_seq(cs * rs, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rs * cs);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_out_seq.size());
    dormidontov_e_highcontrast_mpi::ContrastS ContrastS(taskDataSeq);
    ASSERT_EQ(ContrastS.validation(), true);
    ContrastS.pre_processing();
    ContrastS.run();
    ContrastS.post_processing();

    ASSERT_EQ(res_out_paral, res_out_seq);
  }
}

TEST(dormidontov_e_highcontrast_mpi, Test_just_test_if_it_finally_works2) {
  boost::mpi::communicator world;
  int rs = 2;
  int cs = 2;

  std::vector<int> matrix(cs * rs);
  matrix = dormidontov_e_highcontrast_mpi::generate_halftone_pic(cs, rs);

  std::vector<int> res_out_paral(cs * rs, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(rs * cs);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }
  dormidontov_e_highcontrast_mpi::ContrastP ContrastP(taskDataPar);

  ASSERT_EQ(ContrastP.validation(), true);

  ContrastP.pre_processing();

  ContrastP.run();

  ContrastP.post_processing();
  if (world.rank() == 0) {
    std::vector<int> res_out_seq(cs * rs, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rs * cs);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_out_seq.size());
    dormidontov_e_highcontrast_mpi::ContrastS ContrastS(taskDataSeq);
    ASSERT_EQ(ContrastS.validation(), true);
    ContrastS.pre_processing();
    ContrastS.run();
    ContrastS.post_processing();

    ASSERT_EQ(res_out_paral, res_out_seq);
  }
}

TEST(dormidontov_e_highcontrast_mpi, Test_Empty) {
  boost::mpi::communicator world;
  const int rs = 0;
  const int cs = 0;

  std::vector<int> matrix = {};
  std::vector<int> res_out_paral(cs * rs, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  dormidontov_e_highcontrast_mpi::ContrastP ContrastP(taskDataPar);

  if (world.rank() == 0) {
    // taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(rs * cs);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
    ASSERT_EQ(ContrastP.validation(), false);
  }
}

TEST(dormidontov_e_highcontrast_mpi, Test_just_test_if_it_finally_works5) {
  boost::mpi::communicator world;
  int rs = 2000;
  int cs = 2000;

  std::vector<int> matrix(cs * rs);
  matrix = dormidontov_e_highcontrast_mpi::generate_halftone_pic(cs, rs);
  std::vector<int> res_out_paral(cs * rs, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(rs * cs);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }
  dormidontov_e_highcontrast_mpi::ContrastP ContrastP(taskDataPar);

  ASSERT_EQ(ContrastP.validation(), true);

  ContrastP.pre_processing();

  ContrastP.run();

  ContrastP.post_processing();
  if (world.rank() == 0) {
    std::vector<int> res_out_seq(cs * rs, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rs * cs);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_out_seq.size());
    dormidontov_e_highcontrast_mpi::ContrastS ContrastS(taskDataSeq);
    ASSERT_EQ(ContrastS.validation(), true);
    ContrastS.pre_processing();
    ContrastS.run();
    ContrastS.post_processing();

    ASSERT_EQ(res_out_paral, res_out_seq);
  }
}

TEST(dormidontov_e_highcontrast_mpi, Test_just_test_if_it_finally_works6) {
  boost::mpi::communicator world;
  int rs = 20;
  int cs = 30;

  std::vector<int> matrix(cs * rs);
  matrix = dormidontov_e_highcontrast_mpi::generate_halftone_pic(cs, rs);
  std::vector<int> res_out_paral(cs * rs);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(rs * cs);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }
  dormidontov_e_highcontrast_mpi::ContrastP ContrastP(taskDataPar);
  ASSERT_EQ(ContrastP.validation(), true);
  ContrastP.pre_processing();
  ContrastP.run();
  ContrastP.post_processing();
  if (world.rank() == 0) {
    std::vector<int> res_out_seq(cs * rs, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rs * cs);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_out_seq.size());
    dormidontov_e_highcontrast_mpi::ContrastS ContrastS(taskDataSeq);
    ASSERT_EQ(ContrastS.validation(), true);
    ContrastS.pre_processing();
    ContrastS.run();
    ContrastS.post_processing();

    ASSERT_EQ(res_out_paral, res_out_seq);
  }
}

TEST(dormidontov_e_highcontrast_mpi, Test_just_test_if_it_finally_works7) {
  boost::mpi::communicator world;
  int rs = 14;
  int cs = 88;

  std::vector<int> matrix(cs * rs);
  matrix = dormidontov_e_highcontrast_mpi::generate_halftone_pic(cs, rs);
  std::vector<int> res_out_paral(cs * rs);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(rs * cs);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }
  dormidontov_e_highcontrast_mpi::ContrastP ContrastP(taskDataPar);
  ASSERT_EQ(ContrastP.validation(), true);

  ContrastP.pre_processing();
  ContrastP.run();

  ContrastP.post_processing();
  if (world.rank() == 0) {
    std::vector<int> res_out_seq(cs * rs, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rs * cs);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_out_seq.size());
    dormidontov_e_highcontrast_mpi::ContrastS ContrastS(taskDataSeq);
    ASSERT_EQ(ContrastS.validation(), true);
    ContrastS.pre_processing();
    ContrastS.run();
    ContrastS.post_processing();

    ASSERT_EQ(res_out_paral, res_out_seq);
  }
}

TEST(dormidontov_e_highcontrast_mpi, Test_just_test_if_it_finally_works8) {
  boost::mpi::communicator world;
  int rs = 23;
  int cs = 43;

  std::vector<int> matrix(cs * rs);
  matrix = dormidontov_e_highcontrast_mpi::generate_halftone_pic(cs, rs);
  std::vector<int> res_out_paral(cs * rs);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(rs * cs);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }
  dormidontov_e_highcontrast_mpi::ContrastP ContrastP(taskDataPar);
  ASSERT_EQ(ContrastP.validation(), true);

  ContrastP.pre_processing();
  ContrastP.run();

  ContrastP.post_processing();
  if (world.rank() == 0) {
    std::vector<int> res_out_seq(cs * rs, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rs * cs);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_out_seq.size());
    dormidontov_e_highcontrast_mpi::ContrastS ContrastS(taskDataSeq);
    ASSERT_EQ(ContrastS.validation(), true);
    ContrastS.pre_processing();
    ContrastS.run();
    ContrastS.post_processing();

    ASSERT_EQ(res_out_paral, res_out_seq);
  }
}