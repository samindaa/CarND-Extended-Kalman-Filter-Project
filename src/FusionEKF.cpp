#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
      0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
      0, 0.0009, 0,
      0, 0, 0.09;

  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */

  ekf_.F_ = MatrixXd(4, 4);

  ekf_.F_ << 1, 0, 1, 0,
      0, 1, 0, 1,
      0, 0, 1, 0,
      0, 0, 0, 1;

  ekf_.x_ = VectorXd(4);
  ekf_.Q_ = MatrixXd(ekf_.x_.rows(), ekf_.x_.rows());
  ekf_.P_ = MatrixXd(ekf_.x_.rows(), ekf_.x_.rows());

  H_laser_ << 1, 0, 0, 0,
      0, 1, 0, 0;

  // Start with some covariance
  ekf_.P_ << 1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 10, 0,
      0, 0, 0, 10;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */

      ekf_.x_ << measurement_pack.raw_measurements_[0] * std::cos(measurement_pack.raw_measurements_[1]),
          measurement_pack.raw_measurements_[0] * std::sin(measurement_pack.raw_measurements_[1]),
          0,
          0;

    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      ekf_.x_ << measurement_pack.raw_measurements_[0],
          measurement_pack.raw_measurements_[1],
          0,
          0;
    }

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  const double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;    //dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  //1. Modify the F matrix so that the time is integrated
  Eigen::MatrixXd &F = ekf_.F_;
  F.setIdentity();
  F(0, 2) = F(1, 3) = dt;

  //2. Set the process covariance matrix Q
  const double dt2 = std::pow(dt, 2);
  const double dt3 = dt2 * dt;
  const double dt4 = dt2 * dt2;

  Eigen::MatrixXd &Q = ekf_.Q_;
  Q.setZero(ekf_.x_.rows(), ekf_.x_.rows());

  Q(0, 0) = dt4 * 9.0 / 4.0;
  Q(0, 2) = dt3 * 9.0 / 2.0;

  Q(1, 1) = dt4 * 9.0 / 4.0;
  Q(1, 3) = dt3 * 9.0 / 2.0;

  Q(2, 0) = dt3 * 9.0 / 2.0;
  Q(2, 2) = dt2 * 9.0;

  Q(3, 1) = dt3 * 9.0 / 2.0;
  Q(3, 3) = dt2 * 9.0;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;

    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    Hj_ = ekf_.H_; // Save the current

  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;

    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
