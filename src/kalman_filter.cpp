#include "kalman_filter.h"
#include "tools.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */

  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */

  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  UpdateCommon(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */

  Tools tools;

  H_ = tools.CalculateJacobian(x_);

  //recover state parameters
  const double px = x_(0);
  const double py = x_(1);
  const double vx = x_(2);
  const double vy = x_(3);

  //pre-compute a set of terms to avoid repeated calculation
  double c1 = px * px + py * py;
  if (fabs(c1) < 1e-4) {
    c1 = 1e-4; // some small number and positive
  }
  const double c2 = sqrt(c1);
  
  VectorXd z_pred = VectorXd(3);
  z_pred(0) = c2;
  z_pred(1) = std::atan2(py, px);

  // stability check
  z_pred(2) = (px * vx + py * vy) / c2;
  VectorXd y = z - z_pred;

  auto angle_norm = [](double x) {
    x = fmod(x + M_PI, 2.0 * M_PI);
    if (x < 0)
      x += 2.0 * M_PI;
    return x - M_PI;
  };

  y(1) = angle_norm(y(1));

  UpdateCommon(y);
}

void KalmanFilter::UpdateCommon(const Eigen::VectorXd &y) {
  const MatrixXd Ht = H_.transpose();
  const MatrixXd P_Ht = P_ * Ht;
  const MatrixXd S = H_ * P_Ht + R_;
  const MatrixXd Si = S.inverse();
  const MatrixXd K = P_Ht * Si;

  //new estimate
  x_ = x_ + (K * y);
  const long x_size = x_.size();
  const MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
