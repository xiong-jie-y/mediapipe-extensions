#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <librealsense2/rs.hpp>
#include <mutex>

#include <cmath>


namespace py = pybind11;

struct double3 { 
    double x, y, z; 
    double3 operator*(double t)
    {
        return { x * t, y * t, z * t };
    }

    double3 operator-(double t)
    {
        return { x - t, y - t, z - t };
    }

    void operator*=(double t)
    {
        x = x * t;
        y = y * t;
        z = z * t;
    }

    void operator=(double3 other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
    }

    void add(double t1, double t2, double t3)
    {
        x += t1;
        y += t2;
        z += t3;
    }
};

#ifndef PI
const double PI = 3.14159265358979323846;
#endif

class RotationEstimator {
  // theta is the angle of camera rotation in x, y and z components
  double3 theta;
  std::mutex theta_mtx;
  /* alpha indicates the part that gyro and accelerometer take in computation of
  theta; higher alpha gives more weight to gyro, but too high values cause
  drift; lower alpha gives more weight to accelerometer, which is more sensitive
  to disturbances */
  float alpha_ = 0.7;
  bool first_ = true;
  // Keeps the arrival time of previous gyro frame
  double last_ts_gyro = 0;

 public:
  RotationEstimator(float alpha, bool first) {
      alpha_ = alpha;
      first_ = first;
  }
  // Function to calculate the change in angle of motion based on data from gyro
  void process_gyro(Eigen::Vector3d gyro_data, double ts) {
    if (first_)  // On the first_ iteration, use only data from accelerometer to
                // set the camera's initial position
    {
      last_ts_gyro = ts;
      return;
    }
    // Holds the change in angle, as calculated from gyro
    double3 gyro_angle;

    // Initialize gyro_angle with data from gyro
    gyro_angle.x = gyro_data.x();  // Pitch
    gyro_angle.y = gyro_data.y();  // Yaw
    gyro_angle.z = gyro_data.z();  // Roll

    // Compute the difference between arrival times of previous and current gyro
    // frames
    double dt_gyro = (ts - last_ts_gyro) / 1000.0;
    last_ts_gyro = ts;

    // Change in angle equals gyro measures * time passed since last measurement
    gyro_angle = gyro_angle * dt_gyro;

    // Apply the calculated change of angle to the current angle (theta)
    std::lock_guard<std::mutex> lock(theta_mtx);
    theta.add(-gyro_angle.z, -gyro_angle.y, gyro_angle.x);
  }

  void process_accel(Eigen::Vector3d accel_data) {
    // Holds the angle as calculated from accelerometer data
    double3 accel_angle;

    // Calculate rotation angle from accelerometer data
    accel_angle.z = atan2(accel_data.y(), accel_data.z());
    if (accel_angle.z < 0) {
      accel_angle.z += 2 * PI;
    }
    accel_angle.x = atan2(accel_data.x(), sqrt(accel_data.y() * accel_data.y() +
                                             accel_data.z() * accel_data.z()));
    if (accel_angle.x < 0) {
      accel_angle.x += 2 * PI;
    }

    // If it is the first_ iteration, set initial pose of camera according to
    // accelerometer data (note the different handling for Y axis)
    std::lock_guard<std::mutex> lock(theta_mtx);
    if (first_) {
      first_ = false;
      theta = accel_angle;
      // Since we can't infer the angle around Y axis using accelerometer data,
      // we'll use PI as a convetion for the initial pose
      theta.y = PI;
    } else {
      /*
      Apply Complementary Filter:
          - high-pass filter = theta * alpha_:  allows short-duration signals to
      pass through while filtering out signals that are steady over time, is
      used to cancel out drift.
          - low-pass filter = accel * (1- alpha_): lets through long term
      changes, filtering out short term fluctuations
      */
      theta.x = theta.x * alpha_ + accel_angle.x * (1 - alpha_);
      theta.z = theta.z * alpha_ + accel_angle.z * (1 - alpha_);
    }
  }

  // Returns the current rotation angle
  Eigen::Vector3f get_theta() {
    std::lock_guard<std::mutex> lock(theta_mtx);
    return Eigen::Vector3f(theta.x, theta.y, theta.z);
  }
};

PYBIND11_MODULE(orientation_estimator, m) {
  py::class_<RotationEstimator>(m, "RotationEstimator")
        .def(py::init<float, bool>())
      .def("process_gyro", &RotationEstimator::process_gyro)
      .def("process_accel", &RotationEstimator::process_accel)
      .def("get_theta", &RotationEstimator::get_theta);
}