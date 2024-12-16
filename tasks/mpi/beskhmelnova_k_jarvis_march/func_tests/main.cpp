#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>

#include "mpi/beskhmelnova_k_jarvis_march/include/jarvis_march.hpp"

TEST(beskhmelnova_k_jarvis_march_mpi, Test_empty_triangle) {
  boost::mpi::communicator world;

  // Create data
  int num_points = 3;
  std::vector<double> x;
  std::vector<double> y;

  int hull_size_par;
  std::vector<double> hull_x_par;
  std::vector<double> hull_y_par;

  int hull_size_seq;
  std::vector<double> hull_x_seq;
  std::vector<double> hull_y_seq;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    x = {0.0, 1.0, -1.0};
    y = {1.0, -1.0, -1.0};

    hull_x_par = std::vector<double>(num_points);
    hull_y_par = std::vector<double>(num_points);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
    taskDataPar->inputs_count.emplace_back(x.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
    taskDataPar->inputs_count.emplace_back(y.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x_par.data()));
    taskDataPar->outputs_count.emplace_back(hull_x_par.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y_par.data()));
    taskDataPar->outputs_count.emplace_back(hull_y_par.size());
  }

  beskhmelnova_k_jarvis_march_mpi::TestMPITaskParallel<double> testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    x = {0.0, 1.0, -1.0};
    y = {1.0, -1.0, -1.0};

    hull_x_seq = std::vector<double>(num_points);
    hull_y_seq = std::vector<double>(num_points);

    int res_size = 3;
    std::vector<double> res_x = {-1.0, 1.0, 0.0};
    std::vector<double> res_y = {-1.0, -1.0, 1.0};

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
    taskDataSeq->inputs_count.emplace_back(x.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
    taskDataSeq->inputs_count.emplace_back(y.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x_seq.data()));
    taskDataSeq->outputs_count.emplace_back(hull_x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y_seq.data()));
    taskDataSeq->outputs_count.emplace_back(hull_y_seq.size());

    beskhmelnova_k_jarvis_march_mpi::TestMPITaskSequential<double> testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(hull_size_par, res_size);
    for (int i = 0; i < res_size; i++) {
      ASSERT_EQ(hull_x_par[i], res_x[i]);
      ASSERT_EQ(hull_y_par[i], res_y[i]);
    }

    ASSERT_EQ(hull_size_seq, res_size);
    for (int i = 0; i < res_size; i++) {
      ASSERT_EQ(hull_x_seq[i], res_x[i]);
      ASSERT_EQ(hull_y_seq[i], res_y[i]);
    }
  }
}

TEST(beskhmelnova_k_jarvis_march_mpi, Test_triangle_with_points) {
  boost::mpi::communicator world;

  // Create data
  int num_points = 7;
  std::vector<double> x;
  std::vector<double> y;

  int hull_size_par;
  std::vector<double> hull_x_par;
  std::vector<double> hull_y_par;

  int hull_size_seq;
  std::vector<double> hull_x_seq;
  std::vector<double> hull_y_seq;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    x = {0.0, 1.0, -1.0, 0.0, 0.0, -0.1, 0.11};
    y = {1.0, -1.0, -1.0, 0.0, -0.5, -0.3, 0.11};

    hull_x_par = std::vector<double>(num_points);
    hull_y_par = std::vector<double>(num_points);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
    taskDataPar->inputs_count.emplace_back(x.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
    taskDataPar->inputs_count.emplace_back(y.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x_par.data()));
    taskDataPar->outputs_count.emplace_back(hull_x_par.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y_par.data()));
    taskDataPar->outputs_count.emplace_back(hull_y_par.size());
  }

  beskhmelnova_k_jarvis_march_mpi::TestMPITaskParallel<double> testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    x = {0.0, 1.0, -1.0, 0.0, 0.0, -0.1, 0.11};
    y = {1.0, -1.0, -1.0, 0.0, -0.5, -0.3, 0.11};

    hull_x_seq = std::vector<double>(num_points);
    hull_y_seq = std::vector<double>(num_points);

    int res_size = 3;
    std::vector<double> res_x = {-1.0, 1.0, 0.0};
    std::vector<double> res_y = {-1.0, -1.0, 1.0};

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
    taskDataSeq->inputs_count.emplace_back(x.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
    taskDataSeq->inputs_count.emplace_back(y.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x_seq.data()));
    taskDataSeq->outputs_count.emplace_back(hull_x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y_seq.data()));
    taskDataSeq->outputs_count.emplace_back(hull_y_seq.size());

    beskhmelnova_k_jarvis_march_mpi::TestMPITaskSequential<double> testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(hull_size_par, res_size);
    for (int i = 0; i < res_size; i++) {
      ASSERT_EQ(hull_x_par[i], res_x[i]);
      ASSERT_EQ(hull_y_par[i], res_y[i]);
    }

    ASSERT_EQ(hull_size_seq, res_size);
    for (int i = 0; i < res_size; i++) {
      ASSERT_EQ(hull_x_seq[i], res_x[i]);
      ASSERT_EQ(hull_y_seq[i], res_y[i]);
    }
  }
}

TEST(beskhmelnova_k_jarvis_march_mpi, Test_empty_square) {
  boost::mpi::communicator world;

  // Create data
  int num_points = 4;
  std::vector<double> x;
  std::vector<double> y;

  int hull_size_par;
  std::vector<double> hull_x_par;
  std::vector<double> hull_y_par;

  int hull_size_seq;
  std::vector<double> hull_x_seq;
  std::vector<double> hull_y_seq;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    x = {1.0, -1.0, -1.0, 1.0};
    y = {1.0, 1.0, -1.0, -1.0};

    hull_x_par = std::vector<double>(num_points);
    hull_y_par = std::vector<double>(num_points);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
    taskDataPar->inputs_count.emplace_back(x.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
    taskDataPar->inputs_count.emplace_back(y.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x_par.data()));
    taskDataPar->outputs_count.emplace_back(hull_x_par.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y_par.data()));
    taskDataPar->outputs_count.emplace_back(hull_y_par.size());
  }

  beskhmelnova_k_jarvis_march_mpi::TestMPITaskParallel<double> testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    x = {1.0, -1.0, -1.0, 1.0};
    y = {1.0, 1.0, -1.0, -1.0};

    hull_x_seq = std::vector<double>(num_points);
    hull_y_seq = std::vector<double>(num_points);

    int res_size = 4;
    std::vector<double> res_x = {-1.0, 1.0, 1.0, -1.0};
    std::vector<double> res_y = {-1.0, -1.0, 1.0, 1.0};

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
    taskDataSeq->inputs_count.emplace_back(x.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
    taskDataSeq->inputs_count.emplace_back(y.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x_seq.data()));
    taskDataSeq->outputs_count.emplace_back(hull_x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y_seq.data()));
    taskDataSeq->outputs_count.emplace_back(hull_y_seq.size());

    beskhmelnova_k_jarvis_march_mpi::TestMPITaskSequential<double> testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(hull_size_par, res_size);
    for (int i = 0; i < res_size; i++) {
      ASSERT_EQ(hull_x_par[i], res_x[i]);
      ASSERT_EQ(hull_y_par[i], res_y[i]);
    }

    ASSERT_EQ(hull_size_seq, res_size);
    for (int i = 0; i < res_size; i++) {
      ASSERT_EQ(hull_x_seq[i], res_x[i]);
      ASSERT_EQ(hull_y_seq[i], res_y[i]);
    }
  }
}

TEST(beskhmelnova_k_jarvis_march_mpi, Test_square_with_points) {
  boost::mpi::communicator world;

  // Create data
  int num_points = 7;
  std::vector<double> x;
  std::vector<double> y;

  int hull_size_par;
  std::vector<double> hull_x_par;
  std::vector<double> hull_y_par;

  int hull_size_seq;
  std::vector<double> hull_x_seq;
  std::vector<double> hull_y_seq;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    x = {1.0, 0.0, -1.0, 0.35, -1.0, 1.0, 0.2};
    y = {1.0, 0.0, 1.0, -0.7, -1.0, -1.0, 0.8};

    hull_x_par = std::vector<double>(num_points);
    hull_y_par = std::vector<double>(num_points);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
    taskDataPar->inputs_count.emplace_back(x.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
    taskDataPar->inputs_count.emplace_back(y.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x_par.data()));
    taskDataPar->outputs_count.emplace_back(hull_x_par.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y_par.data()));
    taskDataPar->outputs_count.emplace_back(hull_y_par.size());
  }

  beskhmelnova_k_jarvis_march_mpi::TestMPITaskParallel<double> testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    x = {1.0, 0.0, -1.0, 0.35, -1.0, 1.0, 0.2};
    y = {1.0, 0.0, 1.0, -0.7, -1.0, -1.0, 0.8};

    hull_x_seq = std::vector<double>(num_points);
    hull_y_seq = std::vector<double>(num_points);

    int res_size = 4;
    std::vector<double> res_x = {-1.0, 1.0, 1.0, -1.0};
    std::vector<double> res_y = {-1.0, -1.0, 1.0, 1.0};

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
    taskDataSeq->inputs_count.emplace_back(x.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
    taskDataSeq->inputs_count.emplace_back(y.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x_seq.data()));
    taskDataSeq->outputs_count.emplace_back(hull_x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y_seq.data()));
    taskDataSeq->outputs_count.emplace_back(hull_y_seq.size());

    beskhmelnova_k_jarvis_march_mpi::TestMPITaskSequential<double> testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(hull_size_par, res_size);
    for (int i = 0; i < res_size; i++) {
      ASSERT_EQ(hull_x_par[i], res_x[i]);
      ASSERT_EQ(hull_y_par[i], res_y[i]);
    }

    ASSERT_EQ(hull_size_seq, res_size);
    for (int i = 0; i < res_size; i++) {
      ASSERT_EQ(hull_x_seq[i], res_x[i]);
      ASSERT_EQ(hull_y_seq[i], res_y[i]);
    }
  }
}

TEST(beskhmelnova_k_jarvis_march_mpi, Test_square_with_20_random_points) {
  boost::mpi::communicator world;

  // Create data
  int num_points = 20;
  std::vector<double> x;
  std::vector<double> y;

  int hull_size_par;
  std::vector<double> hull_x_par;
  std::vector<double> hull_y_par;

  int hull_size_seq;
  std::vector<double> hull_x_seq;
  std::vector<double> hull_y_seq;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    x = std::vector<double>(num_points);
    y = std::vector<double>(num_points);
    for (int i = 4; i < num_points; i++) {
      x[i] = std::rand() % 1000;
      y[i] = std::rand() % 1000;
    }

    x[0] = -1.0;
    y[0] = -1.0;

    x[1] = -1.0;
    y[1] = 1000.0;

    x[2] = 1000.0;
    y[2] = 1000.0;

    x[3] = 1000.0;
    y[3] = -1.0;

    hull_x_par = std::vector<double>(num_points);
    hull_y_par = std::vector<double>(num_points);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
    taskDataPar->inputs_count.emplace_back(x.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
    taskDataPar->inputs_count.emplace_back(y.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x_par.data()));
    taskDataPar->outputs_count.emplace_back(hull_x_par.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y_par.data()));
    taskDataPar->outputs_count.emplace_back(hull_y_par.size());
  }

  beskhmelnova_k_jarvis_march_mpi::TestMPITaskParallel<double> testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    x = std::vector<double>(num_points);
    y = std::vector<double>(num_points);
    for (int i = 4; i < num_points; i++) {
      x[i] = std::rand() % 1000;
      y[i] = std::rand() % 1000;
    }

    x[0] = -1.0;
    y[0] = -1.0;

    x[1] = -1.0;
    y[1] = 1000.0;

    x[2] = 1000.0;
    y[2] = 1000.0;

    x[3] = 1000.0;
    y[3] = -1.0;

    hull_x_seq = std::vector<double>(num_points);
    hull_y_seq = std::vector<double>(num_points);

    int res_size = 4;
    std::vector<double> res_x = {-1.0, 1000.0, 1000.0, -1.0};
    std::vector<double> res_y = {-1.0, -1.0, 1000.0, 1000.0};

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
    taskDataSeq->inputs_count.emplace_back(x.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
    taskDataSeq->inputs_count.emplace_back(y.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x_seq.data()));
    taskDataSeq->outputs_count.emplace_back(hull_x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y_seq.data()));
    taskDataSeq->outputs_count.emplace_back(hull_y_seq.size());

    beskhmelnova_k_jarvis_march_mpi::TestMPITaskSequential<double> testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(hull_size_par, res_size);
    for (int i = 0; i < res_size; i++) {
      ASSERT_EQ(hull_x_par[i], res_x[i]);
      ASSERT_EQ(hull_y_par[i], res_y[i]);
    }

    ASSERT_EQ(hull_size_seq, res_size);
    for (int i = 0; i < res_size; i++) {
      ASSERT_EQ(hull_x_seq[i], res_x[i]);
      ASSERT_EQ(hull_y_seq[i], res_y[i]);
    }
  }
}

TEST(beskhmelnova_k_jarvis_march_mpi, Test_square_with_100_random_points) {
  boost::mpi::communicator world;

  // Create data
  int num_points = 100;
  std::vector<double> x;
  std::vector<double> y;

  int hull_size_par;
  std::vector<double> hull_x_par;
  std::vector<double> hull_y_par;

  int hull_size_seq;
  std::vector<double> hull_x_seq;
  std::vector<double> hull_y_seq;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    x = std::vector<double>(num_points);
    y = std::vector<double>(num_points);
    for (int i = 4; i < num_points; i++) {
      x[i] = std::rand() % 1000;
      y[i] = std::rand() % 1000;
    }

    x[0] = -1.0;
    y[0] = -1.0;

    x[1] = -1.0;
    y[1] = 1000.0;

    x[2] = 1000.0;
    y[2] = 1000.0;

    x[3] = 1000.0;
    y[3] = -1.0;

    hull_x_par = std::vector<double>(num_points);
    hull_y_par = std::vector<double>(num_points);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
    taskDataPar->inputs_count.emplace_back(x.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
    taskDataPar->inputs_count.emplace_back(y.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x_par.data()));
    taskDataPar->outputs_count.emplace_back(hull_x_par.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y_par.data()));
    taskDataPar->outputs_count.emplace_back(hull_y_par.size());
  }

  beskhmelnova_k_jarvis_march_mpi::TestMPITaskParallel<double> testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    x = std::vector<double>(num_points);
    y = std::vector<double>(num_points);
    for (int i = 4; i < num_points; i++) {
      x[i] = std::rand() % 1000;
      y[i] = std::rand() % 1000;
    }

    x[0] = -1.0;
    y[0] = -1.0;

    x[1] = -1.0;
    y[1] = 1000.0;

    x[2] = 1000.0;
    y[2] = 1000.0;

    x[3] = 1000.0;
    y[3] = -1.0;

    hull_x_seq = std::vector<double>(num_points);
    hull_y_seq = std::vector<double>(num_points);

    int res_size = 4;
    std::vector<double> res_x = {-1.0, 1000.0, 1000.0, -1.0};
    std::vector<double> res_y = {-1.0, -1.0, 1000.0, 1000.0};

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
    taskDataSeq->inputs_count.emplace_back(x.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
    taskDataSeq->inputs_count.emplace_back(y.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x_seq.data()));
    taskDataSeq->outputs_count.emplace_back(hull_x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y_seq.data()));
    taskDataSeq->outputs_count.emplace_back(hull_y_seq.size());

    beskhmelnova_k_jarvis_march_mpi::TestMPITaskSequential<double> testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(hull_size_par, res_size);
    for (int i = 0; i < res_size; i++) {
      ASSERT_EQ(hull_x_par[i], res_x[i]);
      ASSERT_EQ(hull_y_par[i], res_y[i]);
    }

    ASSERT_EQ(hull_size_seq, res_size);
    for (int i = 0; i < res_size; i++) {
      ASSERT_EQ(hull_x_seq[i], res_x[i]);
      ASSERT_EQ(hull_y_seq[i], res_y[i]);
    }
  }
}

TEST(beskhmelnova_k_jarvis_march_mpi, Test_empty_square_int_points) {
  boost::mpi::communicator world;

  // Create data
  int num_points = 4;
  std::vector<int> x;
  std::vector<int> y;

  int hull_size_par;
  std::vector<int> hull_x_par;
  std::vector<int> hull_y_par;

  int hull_size_seq;
  std::vector<int> hull_x_seq;
  std::vector<int> hull_y_seq;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    x = {1, -1, -1, 1};
    y = {1, 1, -1, -1};

    hull_x_par = std::vector<int>(num_points);
    hull_y_par = std::vector<int>(num_points);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
    taskDataPar->inputs_count.emplace_back(x.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
    taskDataPar->inputs_count.emplace_back(y.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x_par.data()));
    taskDataPar->outputs_count.emplace_back(hull_x_par.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y_par.data()));
    taskDataPar->outputs_count.emplace_back(hull_y_par.size());
  }

  beskhmelnova_k_jarvis_march_mpi::TestMPITaskParallel<int> testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    x = {1, -1, -1, 1};
    y = {1, 1, -1, -1};

    hull_x_seq = std::vector<int>(num_points);
    hull_y_seq = std::vector<int>(num_points);

    int res_size = 4;
    std::vector<int> res_x = {-1, 1, 1, -1};
    std::vector<int> res_y = {-1, -1, 1, 1};

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
    taskDataSeq->inputs_count.emplace_back(x.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
    taskDataSeq->inputs_count.emplace_back(y.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x_seq.data()));
    taskDataSeq->outputs_count.emplace_back(hull_x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y_seq.data()));
    taskDataSeq->outputs_count.emplace_back(hull_y_seq.size());

    beskhmelnova_k_jarvis_march_mpi::TestMPITaskSequential<int> testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(hull_size_par, res_size);
    for (int i = 0; i < res_size; i++) {
      ASSERT_EQ(hull_x_par[i], res_x[i]);
      ASSERT_EQ(hull_y_par[i], res_y[i]);
    }

    ASSERT_EQ(hull_size_seq, res_size);
    for (int i = 0; i < res_size; i++) {
      ASSERT_EQ(hull_x_seq[i], res_x[i]);
      ASSERT_EQ(hull_y_seq[i], res_y[i]);
    }
  }
}

TEST(beskhmelnova_k_jarvis_march_mpi, Test_square_with_100_random_int_points) {
  boost::mpi::communicator world;

  // Create data
  int num_points = 100;
  std::vector<int> x;
  std::vector<int> y;

  int hull_size_par;
  std::vector<int> hull_x_par;
  std::vector<int> hull_y_par;

  int hull_size_seq;
  std::vector<int> hull_x_seq;
  std::vector<int> hull_y_seq;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    x = std::vector<int>(num_points);
    y = std::vector<int>(num_points);
    for (int i = 4; i < num_points; i++) {
      x[i] = std::rand() % 1000;
      y[i] = std::rand() % 1000;
    }

    x[0] = -1;
    y[0] = -1;

    x[1] = -1;
    y[1] = 1000;

    x[2] = 1000;
    y[2] = 1000;

    x[3] = 1000;
    y[3] = -1;

    hull_x_par = std::vector<int>(num_points);
    hull_y_par = std::vector<int>(num_points);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
    taskDataPar->inputs_count.emplace_back(x.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
    taskDataPar->inputs_count.emplace_back(y.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x_par.data()));
    taskDataPar->outputs_count.emplace_back(hull_x_par.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y_par.data()));
    taskDataPar->outputs_count.emplace_back(hull_y_par.size());
  }

  beskhmelnova_k_jarvis_march_mpi::TestMPITaskParallel<int> testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    x = std::vector<int>(num_points);
    y = std::vector<int>(num_points);
    for (int i = 4; i < num_points; i++) {
      x[i] = std::rand() % 1000;
      y[i] = std::rand() % 1000;
    }

    x[0] = -1;
    y[0] = -1;

    x[1] = -1;
    y[1] = 1000;

    x[2] = 1000;
    y[2] = 1000;

    x[3] = 1000;
    y[3] = -1;

    hull_x_seq = std::vector<int>(num_points);
    hull_y_seq = std::vector<int>(num_points);

    int res_size = 4;
    std::vector<int> res_x = {-1, 1000, 1000, -1};
    std::vector<int> res_y = {-1, -1, 1000, 1000};

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
    taskDataSeq->inputs_count.emplace_back(x.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
    taskDataSeq->inputs_count.emplace_back(y.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x_seq.data()));
    taskDataSeq->outputs_count.emplace_back(hull_x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y_seq.data()));
    taskDataSeq->outputs_count.emplace_back(hull_y_seq.size());

    beskhmelnova_k_jarvis_march_mpi::TestMPITaskSequential<int> testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(hull_size_par, res_size);
    for (int i = 0; i < res_size; i++) {
      ASSERT_EQ(hull_x_par[i], res_x[i]);
      ASSERT_EQ(hull_y_par[i], res_y[i]);
    }

    ASSERT_EQ(hull_size_seq, res_size);
    for (int i = 0; i < res_size; i++) {
      ASSERT_EQ(hull_x_seq[i], res_x[i]);
      ASSERT_EQ(hull_y_seq[i], res_y[i]);
    }
  }
}