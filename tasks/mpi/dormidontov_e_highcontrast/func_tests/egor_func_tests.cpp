#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/dormidontov_e_highcontrast/include/egor_include.hpp"

namespace dormidontov_e_highcontrast_mpi {
inline std::vector<int> generate_halftone_pic(int height, int width) {
  std::vector<int> temp(height * width);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      temp[i * width + j] = rand() % 256;
    }
  }
  return temp;
}
}  // namespace dormidontov_e_highcontrast_mpi

TEST(dormidontov_e_highcontrast_mpi, Test_just_test_if_it_finally_works) {
  boost::mpi::communicator world;
  int height = 7;
  int width = 7;

  std::vector<int> picture(width * height);
  picture = dormidontov_e_highcontrast_mpi::generate_halftone_pic(height, width);
  std::vector<int> res_out_paral(width * height, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));
    taskDataPar->inputs_count.emplace_back(height * width);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }
  dormidontov_e_highcontrast_mpi::ContrastP ContrastP(taskDataPar);

  ASSERT_EQ(ContrastP.validation(), true);

  ContrastP.pre_processing();

  ContrastP.run();

  ContrastP.post_processing();
  if (world.rank() == 0) {
    std::vector<int> res_out_seq(width * height, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));
    taskDataSeq->inputs_count.emplace_back(height * width);
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
  int height = 2;
  int width = 2;

  std::vector<int> picture(width * height);
  picture = dormidontov_e_highcontrast_mpi::generate_halftone_pic(height, width);

  std::vector<int> res_out_paral(width * height, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));
    taskDataPar->inputs_count.emplace_back(height * width);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }
  dormidontov_e_highcontrast_mpi::ContrastP ContrastP(taskDataPar);

  ASSERT_EQ(ContrastP.validation(), true);

  ContrastP.pre_processing();

  ContrastP.run();

  ContrastP.post_processing();
  if (world.rank() == 0) {
    std::vector<int> res_out_seq(width * height, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));
    taskDataSeq->inputs_count.emplace_back(height * width);
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
  const int height = 0;
  const int width = 0;

  std::vector<int> picture = {};
  std::vector<int> res_out_paral(width * height, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  dormidontov_e_highcontrast_mpi::ContrastP ContrastP(taskDataPar);

  if (world.rank() == 0) {
    // taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));
    taskDataPar->inputs_count.emplace_back(height * width);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
    ASSERT_EQ(ContrastP.validation(), false);
  }
}

TEST(dormidontov_e_highcontrast_mpi, Test_just_test_if_it_finally_works5) {
  boost::mpi::communicator world;
  int height = 2000;
  int width = 2000;

  std::vector<int> picture(width * height);
  picture = dormidontov_e_highcontrast_mpi::generate_halftone_pic(height, width);
  std::vector<int> res_out_paral(width * height, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));
    taskDataPar->inputs_count.emplace_back(height * width);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }
  dormidontov_e_highcontrast_mpi::ContrastP ContrastP(taskDataPar);

  ASSERT_EQ(ContrastP.validation(), true);

  ContrastP.pre_processing();

  ContrastP.run();

  ContrastP.post_processing();
  if (world.rank() == 0) {
    std::vector<int> res_out_seq(width * height, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));
    taskDataSeq->inputs_count.emplace_back(height * width);
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
  int height = 20;
  int width = 30;

  std::vector<int> picture(width * height);
  picture = dormidontov_e_highcontrast_mpi::generate_halftone_pic(height, width);
  std::vector<int> res_out_paral(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));
    taskDataPar->inputs_count.emplace_back(height * width);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }
  dormidontov_e_highcontrast_mpi::ContrastP ContrastP(taskDataPar);
  ASSERT_EQ(ContrastP.validation(), true);
  ContrastP.pre_processing();
  ContrastP.run();
  ContrastP.post_processing();
  if (world.rank() == 0) {
    std::vector<int> res_out_seq(width * height, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));
    taskDataSeq->inputs_count.emplace_back(height * width);
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
  int height = 14;
  int width = 88;

  std::vector<int> picture(width * height);
  picture = dormidontov_e_highcontrast_mpi::generate_halftone_pic(height, width);
  std::vector<int> res_out_paral(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));
    taskDataPar->inputs_count.emplace_back(height * width);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }
  dormidontov_e_highcontrast_mpi::ContrastP ContrastP(taskDataPar);
  ASSERT_EQ(ContrastP.validation(), true);

  ContrastP.pre_processing();
  ContrastP.run();

  ContrastP.post_processing();
  if (world.rank() == 0) {
    std::vector<int> res_out_seq(width * height, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));
    taskDataSeq->inputs_count.emplace_back(height * width);
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
  int height = 23;
  int width = 43;

  std::vector<int> picture(width * height);
  picture = dormidontov_e_highcontrast_mpi::generate_halftone_pic(height, width);
  std::vector<int> res_out_paral(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));
    taskDataPar->inputs_count.emplace_back(height * width);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }
  dormidontov_e_highcontrast_mpi::ContrastP ContrastP(taskDataPar);
  ASSERT_EQ(ContrastP.validation(), true);

  ContrastP.pre_processing();
  ContrastP.run();

  ContrastP.post_processing();
  if (world.rank() == 0) {
    std::vector<int> res_out_seq(width * height, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));
    taskDataSeq->inputs_count.emplace_back(height * width);
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