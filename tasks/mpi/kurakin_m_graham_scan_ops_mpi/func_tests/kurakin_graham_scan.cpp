#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/kurakin_m_graham_scan_ops_mpi/include/kurakin_graham_scan_ops_mpi.hpp"

TEST(kurakin_m_graham_scan_mpi, Test_shell_rhomb) {
  boost::mpi::communicator world;
  
  int count_point;
  std::vector<double> point_x;
  std::vector<double> point_y;
  
  int scan_size_par;
  std::vector<double> scan_x_par;
  std::vector<double> scan_y_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  count_point = 4;

  if (world.rank() == 0) {
    point_x = {2.0, 0.0, -2.0, 0.0};
    point_y = {0.0, 2.0, 0.0, -2.0};

    scan_x_par = std::vector<double>(count_point);
    scan_y_par = std::vector<double>(count_point);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
    taskDataPar->inputs_count.emplace_back(point_x.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
    taskDataPar->inputs_count.emplace_back(point_y.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_x_par.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_y_par.size());
  }

  kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    point_x = {2.0, 0.0, -2.0, 0.0};
    point_y = {0.0, 2.0, 0.0, -2.0};

    int scan_size_seq;
    std::vector<double> scan_x_seq(count_point);
    std::vector<double> scan_y_seq(count_point);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
    taskDataSeq->inputs_count.emplace_back(point_x.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
    taskDataSeq->inputs_count.emplace_back(point_y.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x_seq.data()));
    taskDataSeq->outputs_count.emplace_back(scan_x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y_seq.data()));
    taskDataSeq->outputs_count.emplace_back(scan_y_seq.size());

    kurakin_m_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    int ans_size = 4;
    std::vector<double> ans_x = {0.0, 2.0, 0.0, -2.0};
    std::vector<double> ans_y = {-2.0, 0.0, 2.0, 0.0};

    ASSERT_EQ(scan_size_par, ans_size);
    for (int i = 0; i < ans_size; i++) {
      ASSERT_EQ(scan_x_par[i], ans_x[i]);
      ASSERT_EQ(scan_y_par[i], ans_y[i]);
    }

    ASSERT_EQ(scan_size_seq, ans_size);
    for (int i = 0; i < ans_size; i++) {
      ASSERT_EQ(scan_x_seq[i], ans_x[i]);
      ASSERT_EQ(scan_y_seq[i], ans_y[i]);
    }
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_shell_rhomb_with_inside_points) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> point_x;
  std::vector<double> point_y;

  int scan_size_par;
  std::vector<double> scan_x_par;
  std::vector<double> scan_y_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  count_point = 17;

  if (world.rank() == 0) {
    point_x = {0.3, 1.0, 2.0, 0.3, 0.0, 0.0, 0.25, -0.25, 0.0, 0.0, -0.25, 0.25, -0.3, -1.0, -2.0, -0.3, 0.1};
    point_y = {-0.25, 0.0, 0.0, 0.25, -2.0, -1.0, -0.3, -0.3, 1.0, 2.0, 0.3, 0.3, 0.25, 0.0, 0.0, -0.25, 0.1};

    scan_x_par = std::vector<double>(count_point);
    scan_y_par = std::vector<double>(count_point);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
    taskDataPar->inputs_count.emplace_back(point_x.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
    taskDataPar->inputs_count.emplace_back(point_y.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_x_par.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_y_par.size());
  }

  kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    point_x = {0.3, 1.0, 2.0, 0.3, 0.0, 0.0, 0.25, -0.25, 0.0, 0.0, -0.25, 0.25, -0.3, -1.0, -2.0, -0.3, 0.1};
    point_y = {-0.25, 0.0, 0.0, 0.25, -2.0, -1.0, -0.3, -0.3, 1.0, 2.0, 0.3, 0.3, 0.25, 0.0, 0.0, -0.25, 0.1};

    int scan_size_seq;
    std::vector<double> scan_x_seq(count_point);
    std::vector<double> scan_y_seq(count_point);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
    taskDataSeq->inputs_count.emplace_back(point_x.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
    taskDataSeq->inputs_count.emplace_back(point_y.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x_seq.data()));
    taskDataSeq->outputs_count.emplace_back(scan_x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y_seq.data()));
    taskDataSeq->outputs_count.emplace_back(scan_y_seq.size());

    kurakin_m_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    int ans_size = 4;
    std::vector<double> ans_x = {0.0, 2.0, 0.0, -2.0};
    std::vector<double> ans_y = {-2.0, 0.0, 2.0, 0.0};

    ASSERT_EQ(scan_size_par, ans_size);
    for (int i = 0; i < ans_size; i++) {
      ASSERT_EQ(scan_x_par[i], ans_x[i]);
      ASSERT_EQ(scan_y_par[i], ans_y[i]);
    }

    ASSERT_EQ(scan_size_seq, ans_size);
    for (int i = 0; i < ans_size; i++) {
      ASSERT_EQ(scan_x_seq[i], ans_x[i]);
      ASSERT_EQ(scan_y_seq[i], ans_y[i]);
    }
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_shell_square) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> point_x;
  std::vector<double> point_y;

  int scan_size_par;
  std::vector<double> scan_x_par;
  std::vector<double> scan_y_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  count_point = 4;

  if (world.rank() == 0) {
    point_x = {2.0, -2.0, -2.0, 2.0};
    point_y = {2.0, 2.0, -2.0, -2.0};

    scan_x_par = std::vector<double>(count_point);
    scan_y_par = std::vector<double>(count_point);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
    taskDataPar->inputs_count.emplace_back(point_x.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
    taskDataPar->inputs_count.emplace_back(point_y.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_x_par.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_y_par.size());
  }

  kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    point_x = {2.0, -2.0, -2.0, 2.0};
    point_y = {2.0, 2.0, -2.0, -2.0};

    int scan_size_seq;
    std::vector<double> scan_x_seq(count_point);
    std::vector<double> scan_y_seq(count_point);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
    taskDataSeq->inputs_count.emplace_back(point_x.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
    taskDataSeq->inputs_count.emplace_back(point_y.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x_seq.data()));
    taskDataSeq->outputs_count.emplace_back(scan_x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y_seq.data()));
    taskDataSeq->outputs_count.emplace_back(scan_y_seq.size());

    kurakin_m_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    int ans_size = 4;
    std::vector<double> ans_x = {2.0, 2.0, -2.0, -2.0};
    std::vector<double> ans_y = {-2.0, 2.0, 2.0, -2.0};

    ASSERT_EQ(scan_size_par, ans_size);
    for (int i = 0; i < ans_size; i++) {
      ASSERT_EQ(scan_x_par[i], ans_x[i]);
      ASSERT_EQ(scan_y_par[i], ans_y[i]);
    }

    ASSERT_EQ(scan_size_seq, ans_size);
    for (int i = 0; i < ans_size; i++) {
      ASSERT_EQ(scan_x_seq[i], ans_x[i]);
      ASSERT_EQ(scan_y_seq[i], ans_y[i]);
    }
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_shell_square_with_inside_points) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> point_x;
  std::vector<double> point_y;

  int scan_size_par;
  std::vector<double> scan_x_par;
  std::vector<double> scan_y_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  count_point = 17;

  if (world.rank() == 0) {
    point_x = {-2.0, -1.0, -0.5, -1.0, 2.0, 0.5, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0, -2.0, -0.5, -1.0, -1.0, 0.1};
    point_y = {-2.0, -1.0, -1.0, -0.5, -2.0, -1.0, -1.0, -0.5, 2.0, 1.0, 1.0, 0.5, 2.0, 1.0, 1.0, 0.5, 0.1};

    scan_x_par = std::vector<double>(count_point);
    scan_y_par = std::vector<double>(count_point);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
    taskDataPar->inputs_count.emplace_back(point_x.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
    taskDataPar->inputs_count.emplace_back(point_y.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_x_par.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_y_par.size());
  }

  kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    point_x = {-2.0, -1.0, -0.5, -1.0, 2.0, 0.5, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0, -2.0, -0.5, -1.0, -1.0, 0.1};
    point_y = {-2.0, -1.0, -1.0, -0.5, -2.0, -1.0, -1.0, -0.5, 2.0, 1.0, 1.0, 0.5, 2.0, 1.0, 1.0, 0.5, 0.1};

    int scan_size_seq;
    std::vector<double> scan_x_seq(count_point);
    std::vector<double> scan_y_seq(count_point);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
    taskDataSeq->inputs_count.emplace_back(point_x.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
    taskDataSeq->inputs_count.emplace_back(point_y.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x_seq.data()));
    taskDataSeq->outputs_count.emplace_back(scan_x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y_seq.data()));
    taskDataSeq->outputs_count.emplace_back(scan_y_seq.size());

    kurakin_m_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    int ans_size = 4;
    std::vector<double> ans_x = {2.0, 2.0, -2.0, -2.0};
    std::vector<double> ans_y = {-2.0, 2.0, 2.0, -2.0};

    ASSERT_EQ(scan_size_par, ans_size);
    for (int i = 0; i < ans_size; i++) {
      ASSERT_EQ(scan_x_par[i], ans_x[i]);
      ASSERT_EQ(scan_y_par[i], ans_y[i]);
    }

    ASSERT_EQ(scan_size_seq, ans_size);
    for (int i = 0; i < ans_size; i++) {
      ASSERT_EQ(scan_x_seq[i], ans_x[i]);
      ASSERT_EQ(scan_y_seq[i], ans_y[i]);
    }
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_shell_random) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> point_x;
  std::vector<double> point_y;

  int scan_size_par;
  std::vector<double> scan_x_par;
  std::vector<double> scan_y_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  count_point = 100;

  if (world.rank() == 0) {
    kurakin_m_graham_scan_mpi::getRandomVectorForGrahamScan(point_x, point_y, count_point, world.size());

    scan_x_par = std::vector<double>(count_point);
    scan_y_par = std::vector<double>(count_point);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
    taskDataPar->inputs_count.emplace_back(point_x.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
    taskDataPar->inputs_count.emplace_back(point_y.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_x_par.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_y_par.size());
  }

  kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    int scan_size_seq;
    std::vector<double> scan_x_seq(count_point);
    std::vector<double> scan_y_seq(count_point);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
    taskDataSeq->inputs_count.emplace_back(point_x.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
    taskDataSeq->inputs_count.emplace_back(point_y.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x_seq.data()));
    taskDataSeq->outputs_count.emplace_back(scan_x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y_seq.data()));
    taskDataSeq->outputs_count.emplace_back(scan_y_seq.size());

    kurakin_m_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(scan_size_seq, scan_size_par);
    for (int i = 0; i < scan_size_seq; i++) {
      ASSERT_EQ(scan_x_seq[i], scan_x_par[i]);
      ASSERT_EQ(scan_y_seq[i], scan_y_par[i]);
    }
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_validation_count_points) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> point_x;
  std::vector<double> point_y;

  int scan_size_par;
  std::vector<double> scan_x_par;
  std::vector<double> scan_y_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    count_point = 2;
    point_x = {2.0, 1.0};
    point_y = {2.0, 1.0};

    scan_x_par = std::vector<double>(count_point);
    scan_y_par = std::vector<double>(count_point);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
    taskDataPar->inputs_count.emplace_back(point_x.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
    taskDataPar->inputs_count.emplace_back(point_y.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_x_par.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_y_par.size());

    kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_validation_inputs_point) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> point_x;
  std::vector<double> point_y;

  int scan_size_par;
  std::vector<double> scan_x_par;
  std::vector<double> scan_y_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    count_point = 4;
    point_x = {2.0, 1.0, -2.0};
    point_y = {2.0, 1.0, 2.0, 1.0};

    scan_x_par = std::vector<double>(count_point);
    scan_y_par = std::vector<double>(count_point);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
    taskDataPar->inputs_count.emplace_back(point_x.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
    taskDataPar->inputs_count.emplace_back(point_y.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_x_par.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_y_par.size());

    kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_validation_outputs_point) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> point_x;
  std::vector<double> point_y;

  int scan_size_par;
  std::vector<double> scan_x_par;
  std::vector<double> scan_y_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    count_point = 4;
    point_x = {2.0, 1.0, -2.0};
    point_y = {2.0, 1.0, 2.0, 1.0};

    scan_x_par = std::vector<double>(count_point - 1);
    scan_y_par = std::vector<double>(count_point);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
    taskDataPar->inputs_count.emplace_back(point_x.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
    taskDataPar->inputs_count.emplace_back(point_y.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_x_par.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_y_par.size());

    kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_validation_inputs_count) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> point_x;

  int scan_size_par;
  std::vector<double> scan_x_par;
  std::vector<double> scan_y_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    count_point = 4;
    point_x = {2.0, 1.0, -2.0, -1.0};

    scan_x_par = std::vector<double>(count_point);
    scan_y_par = std::vector<double>(count_point);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
    taskDataPar->inputs_count.emplace_back(point_x.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_x_par.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_y_par.size());

    kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_validation_outputs_count) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> point_x;
  std::vector<double> point_y;

  int scan_size_par;
  std::vector<double> scan_x_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    count_point = 4;
    point_x = {2.0, 1.0, -2.0, -1.0};

    scan_x_par = std::vector<double>(count_point);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
    taskDataPar->inputs_count.emplace_back(point_x.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
    taskDataPar->inputs_count.emplace_back(point_y.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_x_par.size());

    kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}
