// Filatev Vladislav Metod Belmana Forda
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <queue>
#include <random>
#include <vector>

#include "mpi/filatev_v_metod_belmana_forda/include/ops_mpi.hpp"

TEST(filatev_v_metod_belmana_forda_mpi, test_simpel_path) {
  boost::mpi::communicator world;
  int n = 6;
  int m = 10;
  int start = 0;
  std::vector<int> Adjncy;
  std::vector<int> Xadj;
  std::vector<int> Eweights;
  std::vector<int> d;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    Adjncy = {1, 2, 3, 4, 1, 4, 5, 4, 5, 5};
    Xadj = {0, 2, 4, 7, 9, 10, 10};
    Eweights = {7, 9, -1, -2, -3, 2, 1, 1, 3, 3};
    d.resize(n);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
    taskData->inputs_count.emplace_back(n);
    taskData->inputs_count.emplace_back(m);
    taskData->inputs_count.emplace_back(start);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(d.data()));
    taskData->outputs_count.emplace_back(n);
  }

  filatev_v_metod_belmana_forda_mpi::MetodBelmanaFordaMPI metodBelmanaForda(taskData);

  ASSERT_TRUE(metodBelmanaForda.validation());
  metodBelmanaForda.pre_processing();
  metodBelmanaForda.run();
  // metodBelmanaForda.post_processing();

  // if (world.rank() == 0) {
  //   std::vector<int> tResh = {0, 6, 9, 5, 4, 7};

  //   ASSERT_EQ(tResh, d);
  // }
}

// TEST(filatev_v_metod_belmana_forda_mpi, test_simpel_path2) {
//   boost::mpi::communicator world;
//   int n = 4;
//   int m = 12;
//   int start = 0;
//   std::vector<int> Adjncy;
//   std::vector<int> Xadj;
//   std::vector<int> Eweights;
//   std::vector<int> d;

//   std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

//   if (world.rank() == 0) {
//     Adjncy = {1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2};
//     Xadj = {0, 3, 6, 9, 12, 12};
//     Eweights.assign(m, 1);
//     d.resize(n);

//     taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
//     taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
//     taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
//     taskData->inputs_count.emplace_back(n);
//     taskData->inputs_count.emplace_back(m);
//     taskData->inputs_count.emplace_back(start);
//     taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(d.data()));
//     taskData->outputs_count.emplace_back(n);
//   }

//   filatev_v_metod_belmana_forda_mpi::MetodBelmanaFordaMPI metodBelmanaForda(taskData);

//   ASSERT_TRUE(metodBelmanaForda.validation());
//   metodBelmanaForda.pre_processing();
//   metodBelmanaForda.run();
//   metodBelmanaForda.post_processing();

//   if (world.rank() == 0) {
//     std::vector<int> tResh = {0, 1, 1, 1};
//     ASSERT_EQ(tResh, d);
//   }
// }

// TEST(filatev_v_metod_belmana_forda_mpi, test_simpel_path3) {
//   boost::mpi::communicator world;
//   int n = 7;
//   int m = 12;
//   int start = 0;
//   std::vector<int> Adjncy;
//   std::vector<int> Xadj;
//   std::vector<int> Eweights;
//   std::vector<int> d;

//   std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

//   if (world.rank() == 0) {
//     Adjncy = {1, 2, 3, 6, 1, 4, 5, 4, 3, 0, 6, 0};
//     Xadj = {0, 2, 4, 7, 8, 9, 11, 12};
//     Eweights = {8, 5, 1, 8, 2, 1, 3, 2, 1, 5, 2, 4};
//     d.resize(n);

//     taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
//     taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
//     taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
//     taskData->inputs_count.emplace_back(n);
//     taskData->inputs_count.emplace_back(m);
//     taskData->inputs_count.emplace_back(start);
//     taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(d.data()));
//     taskData->outputs_count.emplace_back(n);
//   }

//   filatev_v_metod_belmana_forda_mpi::MetodBelmanaFordaMPI metodBelmanaForda(taskData);

//   ASSERT_TRUE(metodBelmanaForda.validation());
//   metodBelmanaForda.pre_processing();
//   metodBelmanaForda.run();
//   metodBelmanaForda.post_processing();

//   if (world.rank() == 0) {
//     std::vector<int> tResh = {0, 7, 5, 7, 6, 8, 10};
//     ASSERT_EQ(tResh, d);
//   }
// }

// TEST(filatev_v_metod_belmana_forda_mpi, test_error) {
//   boost::mpi::communicator world;
//   int n = 7;
//   int start = 0;
//   std::vector<int> Adjncy;
//   std::vector<int> Xadj;
//   std::vector<int> Eweights;
//   std::vector<int> d;

//   std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

//   if (world.rank() == 0) {
//     Adjncy = {1, 2, 3, 6, 1, 4, 5, 4, 3, 0, 6, 0};
//     Xadj = {0, 2, 4, 7, 8, 9, 11, 12};
//     Eweights = {8, 5, 1, 8, 2, 1, 3, 2, 1, 5, 2, 4};
//     d.resize(n);

//     taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
//     taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
//     taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
//     taskData->inputs_count.emplace_back(0);
//     taskData->inputs_count.emplace_back(0);
//     taskData->inputs_count.emplace_back(start);
//     taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(d.data()));
//     taskData->outputs_count.emplace_back(n);
//   }

//   filatev_v_metod_belmana_forda_mpi::MetodBelmanaFordaMPI metodBelmanaForda(taskData);

//   if (world.rank() == 0) {
//     ASSERT_FALSE(metodBelmanaForda.validation());
//   }
// }

// TEST(filatev_v_metod_belmana_forda_mpi, test_error_2) {
//   boost::mpi::communicator world;
//   int n = 7;
//   int start = 0;
//   std::vector<int> Adjncy;
//   std::vector<int> Xadj;
//   std::vector<int> Eweights;
//   std::vector<int> d;

//   std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

//   if (world.rank() == 0) {
//     Adjncy = {1, 2, 3, 6, 1, 4, 5, 4, 3, 0, 6, 0};
//     Xadj = {0, 2, 4, 7, 8, 9, 11, 12};
//     Eweights = {8, 5, 1, 8, 2, 1, 3, 2, 1, 5, 2, 4};
//     d.resize(n);

//     taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
//     taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
//     taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
//     taskData->inputs_count.emplace_back(n);
//     taskData->inputs_count.emplace_back(54);
//     taskData->inputs_count.emplace_back(start);
//     taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(d.data()));
//     taskData->outputs_count.emplace_back(n);
//   }

//   filatev_v_metod_belmana_forda_mpi::MetodBelmanaFordaMPI metodBelmanaForda(taskData);

//   if (world.rank() == 0) {
//     ASSERT_FALSE(metodBelmanaForda.validation());
//   }
// }

// TEST(filatev_v_metod_belmana_forda_mpi, test_error_3) {
//   boost::mpi::communicator world;
//   int n = 7;
//   int m = 10;
//   int start = 0;
//   std::vector<int> Adjncy;
//   std::vector<int> Xadj;
//   std::vector<int> Eweights;
//   std::vector<int> d;

//   std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

//   if (world.rank() == 0) {
//     Adjncy = {1, 2, 3, 6, 1, 4, 5, 4, 3, 0, 6, 0};
//     Xadj = {0, 2, 4, 7, 8, 9, 11, 12};
//     Eweights = {8, 5, 1, 8, 2, 1, 3, 2, 1, 5, 2, 4};
//     d.resize(n);

//     taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
//     taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
//     taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
//     taskData->inputs_count.emplace_back(n);
//     taskData->inputs_count.emplace_back(m);
//     taskData->inputs_count.emplace_back(start);
//     taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(d.data()));
//     taskData->outputs_count.emplace_back(n + 1);
//   }

//   filatev_v_metod_belmana_forda_mpi::MetodBelmanaFordaMPI metodBelmanaForda(taskData);

//   if (world.rank() == 0) {
//     ASSERT_FALSE(metodBelmanaForda.validation());
//   }
// }

// TEST(filatev_v_metod_belmana_forda_mpi, test_error_4) {
//   boost::mpi::communicator world;
//   int n = 7;
//   int m = 10;
//   std::vector<int> Adjncy;
//   std::vector<int> Xadj;
//   std::vector<int> Eweights;
//   std::vector<int> d;

//   std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

//   if (world.rank() == 0) {
//     Adjncy = {1, 2, 3, 6, 1, 4, 5, 4, 3, 0, 6, 0};
//     Xadj = {0, 2, 4, 7, 8, 9, 11, 12};
//     Eweights = {8, 5, 1, 8, 2, 1, 3, 2, 1, 5, 2, 4};
//     d.resize(n);

//     taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
//     taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
//     taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
//     taskData->inputs_count.emplace_back(n);
//     taskData->inputs_count.emplace_back(m);
//     taskData->inputs_count.emplace_back(25);
//     taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(d.data()));
//     taskData->outputs_count.emplace_back(n);
//   }

//   filatev_v_metod_belmana_forda_mpi::MetodBelmanaFordaMPI metodBelmanaForda(taskData);

//   if (world.rank() == 0) {
//     ASSERT_FALSE(metodBelmanaForda.validation());
//   }
// }