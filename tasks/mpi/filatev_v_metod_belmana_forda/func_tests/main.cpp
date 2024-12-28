// Filatev Vladislav Metod Belmana Forda
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/filatev_v_metod_belmana_forda/include/ops_mpi.hpp"

namespace filatev_v_metod_belmana_forda_mpi {

bool GeneratGraf(int n, int m, std::vector<int> &Adjncy, std::vector<int> &Xadj, std::vector<int> &Eweights) {
  int max = 10;
  int min = -10;
  int kol_m = 0;
  int inf = std::numeric_limits<int>::max();

  if (m > n * n) return false;

  std::vector<std::vector<int>> matrix(n, std::vector<int>(n, inf));
  while (kol_m != m) {
    int j = rand() % n;
    int i = rand() % n;
    if (i != j && matrix[i][j] == inf) {
      matrix[i][j] = rand() % (max - min + 1) + min;
      kol_m++;
    }
  }

  Adjncy.resize(m, 0);
  Xadj.resize(n + 1);
  Eweights.resize(m, 0);

  for (int i = 0, k = 0; i < n; i++) {
    Xadj[i] = k;
    for (int j = 0; j < n; j++) {
      if (i != j && matrix[i][j] != inf) {
        Adjncy[k] = j;
        Eweights[k] = matrix[i][j];
        k++;
      }
    }
  }
  Xadj[n] = m;
  return true;
}

}  // namespace filatev_v_metod_belmana_forda_mpi

TEST(filatev_v_metod_belmana_forda_mpi, test_simpel_path) {
  boost::mpi::communicator world;
  int n = 6;
  int m = 9;
  int start = 0;
  std::vector<int> Adjncy;
  std::vector<int> Xadj;
  std::vector<int> Eweights;
  std::vector<int> d;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    Adjncy = {1, 2, 3, 4, 1, 4, 4, 5, 5};
    Xadj = {0, 2, 4, 6, 8, 9, 9};
    Eweights = {7, 9, -1, -2, -3, 2, 1, 3, 3};
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
  metodBelmanaForda.post_processing();

  if (world.rank() == 0) {
    std::vector<int> tResh = {0, 6, 9, 5, 4, 7};

    ASSERT_EQ(tResh, d);
  }
}

TEST(filatev_v_metod_belmana_forda_mpi, test_simpel_path2) {
  boost::mpi::communicator world;
  int n = 4;
  int m = 12;
  int start = 0;
  std::vector<int> Adjncy;
  std::vector<int> Xadj;
  std::vector<int> Eweights;
  std::vector<int> d;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    Adjncy = {1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2};
    Xadj = {0, 3, 6, 9, 12, 12};
    Eweights.assign(m, 1);
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
  metodBelmanaForda.post_processing();

  if (world.rank() == 0) {
    std::vector<int> tResh = {0, 1, 1, 1};
    ASSERT_EQ(tResh, d);
  }
}

TEST(filatev_v_metod_belmana_forda_mpi, test_simpel_path3) {
  boost::mpi::communicator world;
  int n = 7;
  int m = 12;
  int start = 0;
  std::vector<int> Adjncy;
  std::vector<int> Xadj;
  std::vector<int> Eweights;
  std::vector<int> d;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    Adjncy = {1, 2, 3, 6, 1, 4, 5, 4, 3, 0, 6, 0};
    Xadj = {0, 2, 4, 7, 8, 9, 11, 12};
    Eweights = {8, 5, 1, 8, 2, 1, 3, 2, 1, 5, 2, 4};
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
  metodBelmanaForda.post_processing();

  if (world.rank() == 0) {
    std::vector<int> tResh = {0, 7, 5, 7, 6, 8, 10};
    ASSERT_EQ(tResh, d);
  }
}

TEST(filatev_v_metod_belmana_forda_mpi, test_simpel_path4) {
  boost::mpi::communicator world;
  int inf = std::numeric_limits<int>::max();
  int n = 6;
  int m = 9;
  int start = 1;
  std::vector<int> Adjncy;
  std::vector<int> Xadj;
  std::vector<int> Eweights;
  std::vector<int> d;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    Adjncy = {1, 2, 3, 4, 1, 4, 4, 5, 5};
    Xadj = {0, 2, 4, 6, 8, 9, 9};
    Eweights = {7, 9, -1, -2, -3, 2, 1, 3, 3};
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
  metodBelmanaForda.post_processing();

  if (world.rank() == 0) {
    std::vector<int> tResh = {inf, 0, inf, -1, -2, 1};

    ASSERT_EQ(tResh, d);
  }
}

TEST(filatev_v_metod_belmana_forda_mpi, test_error) {
  boost::mpi::communicator world;
  int n = 7;
  int start = 0;
  std::vector<int> Adjncy;
  std::vector<int> Xadj;
  std::vector<int> Eweights;
  std::vector<int> d;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    Adjncy = {1, 2, 3, 6, 1, 4, 5, 4, 3, 0, 6, 0};
    Xadj = {0, 2, 4, 7, 8, 9, 11, 12};
    Eweights = {8, 5, 1, 8, 2, 1, 3, 2, 1, 5, 2, 4};
    d.resize(n);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
    taskData->inputs_count.emplace_back(0);
    taskData->inputs_count.emplace_back(0);
    taskData->inputs_count.emplace_back(start);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(d.data()));
    taskData->outputs_count.emplace_back(n);
  }

  filatev_v_metod_belmana_forda_mpi::MetodBelmanaFordaMPI metodBelmanaForda(taskData);

  if (world.rank() == 0) {
    ASSERT_FALSE(metodBelmanaForda.validation());
  }
}

TEST(filatev_v_metod_belmana_forda_mpi, test_error_2) {
  boost::mpi::communicator world;
  int n = 7;
  int start = 0;
  std::vector<int> Adjncy;
  std::vector<int> Xadj;
  std::vector<int> Eweights;
  std::vector<int> d;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    Adjncy = {1, 2, 3, 6, 1, 4, 5, 4, 3, 0, 6, 0};
    Xadj = {0, 2, 4, 7, 8, 9, 11, 12};
    Eweights = {8, 5, 1, 8, 2, 1, 3, 2, 1, 5, 2, 4};
    d.resize(n);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
    taskData->inputs_count.emplace_back(n);
    taskData->inputs_count.emplace_back(54);
    taskData->inputs_count.emplace_back(start);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(d.data()));
    taskData->outputs_count.emplace_back(n);
  }

  filatev_v_metod_belmana_forda_mpi::MetodBelmanaFordaMPI metodBelmanaForda(taskData);

  if (world.rank() == 0) {
    ASSERT_FALSE(metodBelmanaForda.validation());
  }
}

TEST(filatev_v_metod_belmana_forda_mpi, test_error_3) {
  boost::mpi::communicator world;
  int n = 7;
  int m = 10;
  int start = 0;
  std::vector<int> Adjncy;
  std::vector<int> Xadj;
  std::vector<int> Eweights;
  std::vector<int> d;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    Adjncy = {1, 2, 3, 6, 1, 4, 5, 4, 3, 0, 6, 0};
    Xadj = {0, 2, 4, 7, 8, 9, 11, 12};
    Eweights = {8, 5, 1, 8, 2, 1, 3, 2, 1, 5, 2, 4};
    d.resize(n);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
    taskData->inputs_count.emplace_back(n);
    taskData->inputs_count.emplace_back(m);
    taskData->inputs_count.emplace_back(start);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(d.data()));
    taskData->outputs_count.emplace_back(n + 1);
  }

  filatev_v_metod_belmana_forda_mpi::MetodBelmanaFordaMPI metodBelmanaForda(taskData);

  if (world.rank() == 0) {
    ASSERT_FALSE(metodBelmanaForda.validation());
  }
}

TEST(filatev_v_metod_belmana_forda_mpi, test_error_4) {
  boost::mpi::communicator world;
  int n = 7;
  int m = 10;
  std::vector<int> Adjncy;
  std::vector<int> Xadj;
  std::vector<int> Eweights;
  std::vector<int> d;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    Adjncy = {1, 2, 3, 6, 1, 4, 5, 4, 3, 0, 6, 0};
    Xadj = {0, 2, 4, 7, 8, 9, 11, 12};
    Eweights = {8, 5, 1, 8, 2, 1, 3, 2, 1, 5, 2, 4};
    d.resize(n);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
    taskData->inputs_count.emplace_back(n);
    taskData->inputs_count.emplace_back(m);
    taskData->inputs_count.emplace_back(25);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(d.data()));
    taskData->outputs_count.emplace_back(n);
  }

  filatev_v_metod_belmana_forda_mpi::MetodBelmanaFordaMPI metodBelmanaForda(taskData);

  if (world.rank() == 0) {
    ASSERT_FALSE(metodBelmanaForda.validation());
  }
}

TEST(filatev_v_metod_belmana_forda_mpi, test_random_path) {
  boost::mpi::communicator world;
  int n = 20;
  int m = 30;
  int start = 0;
  std::vector<int> Adjncy;
  std::vector<int> Xadj;
  std::vector<int> Eweights;
  std::vector<int> d;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    filatev_v_metod_belmana_forda_mpi::GeneratGraf(n, m, Adjncy, Xadj, Eweights);
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
  metodBelmanaForda.post_processing();

  if (world.rank() == 0) {
    std::vector<int> d_s(n);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(start);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(d_s.data()));
    taskDataSeq->outputs_count.emplace_back(n);

    filatev_v_metod_belmana_forda_mpi::MetodBelmanaFordaSeq metodBelmanaFordaSeq(taskDataSeq);

    ASSERT_TRUE(metodBelmanaFordaSeq.validation());
    metodBelmanaFordaSeq.pre_processing();
    metodBelmanaFordaSeq.run();
    metodBelmanaFordaSeq.post_processing();

    ASSERT_EQ(d_s, d);
  }
}

TEST(filatev_v_metod_belmana_forda_mpi, test_random_path2) {
  boost::mpi::communicator world;
  int n = 10;
  int m = 50;
  int start = 0;
  std::vector<int> Adjncy;
  std::vector<int> Xadj;
  std::vector<int> Eweights;
  std::vector<int> d;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    filatev_v_metod_belmana_forda_mpi::GeneratGraf(n, m, Adjncy, Xadj, Eweights);
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
  metodBelmanaForda.post_processing();

  if (world.rank() == 0) {
    std::vector<int> d_s(n);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(start);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(d_s.data()));
    taskDataSeq->outputs_count.emplace_back(n);

    filatev_v_metod_belmana_forda_mpi::MetodBelmanaFordaSeq metodBelmanaFordaSeq(taskDataSeq);

    ASSERT_TRUE(metodBelmanaFordaSeq.validation());
    metodBelmanaFordaSeq.pre_processing();
    metodBelmanaFordaSeq.run();
    metodBelmanaFordaSeq.post_processing();

    ASSERT_EQ(d_s, d);
  }
}