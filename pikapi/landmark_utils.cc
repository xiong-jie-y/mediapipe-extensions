#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "glog/logging.h"

namespace py = pybind11;

std::tuple<Eigen::Vector3d, double> GetRotationVector(Eigen::Ref<Eigen::Vector3d> v1, Eigen::Ref<Eigen::Vector3d> v2) {
    auto rotationVector = Eigen::AngleAxisd(Eigen::Quaterniond::FromTwoVectors(v1, v2));
    return std::make_tuple(rotationVector.axis(), rotationVector.angle());
}

// The input 3D points are stored as columns.
Eigen::Affine3d Find3DAffineTransform(Eigen::Matrix3Xd in, Eigen::Matrix3Xd out) {
    // Default output
    Eigen::Affine3d A;
    A.linear() = Eigen::Matrix3d::Identity(3, 3);
    A.translation() = Eigen::Vector3d::Zero();

    if (in.cols() != out.cols()) throw "Find3DAffineTransform(): input data mis-match";

    // First find the scale, by finding the ratio of sums of some distances,
    // then bring the datasets to the same scale.
    double dist_in = 0, dist_out = 0;
    for (int col = 0; col < in.cols() - 1; col++) {
        dist_in += (in.col(col + 1) - in.col(col)).norm();
        dist_out += (out.col(col + 1) - out.col(col)).norm();
    }
    if (dist_in <= 0 || dist_out <= 0) return A;
    double scale = dist_out / dist_in;
    out /= scale;

    // Find the centroids then shift to the origin
    Eigen::Vector3d in_ctr = Eigen::Vector3d::Zero();
    Eigen::Vector3d out_ctr = Eigen::Vector3d::Zero();
    for (int col = 0; col < in.cols(); col++) {
        in_ctr += in.col(col);
        out_ctr += out.col(col);
    }
    in_ctr /= in.cols();
    out_ctr /= out.cols();
    for (int col = 0; col < in.cols(); col++) {
        in.col(col) -= in_ctr;
        out.col(col) -= out_ctr;
    }

    // SVD
    Eigen::MatrixXd Cov = in * out.transpose();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Cov, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Find the rotation
    double d = (svd.matrixV() * svd.matrixU().transpose()).determinant();
    if (d > 0)
        d = 1.0;
    else
        d = -1.0;
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity(3, 3);
    I(2, 2) = d;
    Eigen::Matrix3d R = svd.matrixV() * I * svd.matrixU().transpose();

    // The final transform
    A.linear() = scale * R;
    A.translation() = scale * (out_ctr - R * in_ctr);

    return A;
}

Eigen::AngleAxisd EstimatePalmAngleFromBase(const std::vector<Eigen::Vector3d>& landmark_list,
                                            const Eigen::MatrixXd& base) {
    // Calculate a maybe stable vectors that penetrating palm perpendicularly.
    Eigen::Vector3d rotationVectorSum;
    auto PALM_PLAIN_INDICES = {5, 9, 13, 17};

    for (size_t i = 0; i < PALM_PLAIN_INDICES.size() - 1; i++) {
        auto a = landmark_list[i] - landmark_list[0];
        auto b = landmark_list[i + 1] - landmark_list[0];
        rotationVectorSum += Eigen::AngleAxisd(Eigen::Quaterniond::FromTwoVectors(a, b)).axis();
    }
    auto meanPalmVector = rotationVectorSum / (PALM_PLAIN_INDICES.size() - 1);

    // Calculate unit direction vectors perpendicular to palm vector.
    Eigen::Vector3d fingerVector = ((landmark_list[9] - landmark_list[0]) + (landmark_list[13] - landmark_list[0]))/2.0;
    fingerVector /= fingerVector.norm();
    Eigen::Vector3d thumbVector = fingerVector.cross(meanPalmVector);
    thumbVector /= thumbVector.norm();

    // Calculate rotation vector that rotate axis.
    Eigen::MatrixXd vectors(3,3);
    vectors.col(0) = thumbVector;
    vectors.col(1) = fingerVector;
    vectors.col(2) = meanPalmVector;

    // std::cout << vectors << std::endl;
    const auto& A = Find3DAffineTransform(base, vectors);
    auto rotation = Eigen::AngleAxisd(A.linear());
    return std::move(rotation);
}


std::tuple<Eigen::Vector3d, double> EstimatePalmRotation(
    const std::vector<Eigen::Vector3d>& landmark_list,
    const std::string& direction) {
    Eigen::Matrix3d baseMatrix;
    // This base is 
    // (1) Fingers ar pointing to camera.
    // (2) 
    if (direction == "Right") {
        baseMatrix << 1, 0, 0,
                    0, -1, 0,
                    0, 0, -1;
    } else if (direction == "Left") {
        baseMatrix << 1, 0, 0,
                    0, -1, 0,
                    0, 0, 1;   
    }        
    auto rotation = EstimatePalmAngleFromBase(landmark_list, baseMatrix);
    return std::make_tuple(rotation.axis(), rotation.angle());
}

std::vector<std::tuple<Eigen::Vector3d, double>> GetRelativeAnglesFromXYPlane(
    const std::vector<Eigen::Vector3d>& landmarkList, const std::vector<int>& ids
) {
    std::vector<Eigen::Vector3d> positions;
    for (auto id : ids) {
        positions.push_back(landmarkList[id]);
    }
    
    // Get the direction of the finger.
    Eigen::Vector3d base = positions.back() - positions.front();
    base[2] = 0;

    std::vector<Eigen::Vector3d> finger_diffs = { base };
    for (size_t i = 0; i < positions.size() - 1; i++)
    {
        finger_diffs.push_back(positions[i + 1] - positions[i]);
    }

    std::vector<std::tuple<Eigen::Vector3d, double>> rotations;
    for (size_t i = 0; i < finger_diffs.size() - 1; i++)
    {
        rotations.push_back(GetRotationVector(finger_diffs[i + 1], finger_diffs[i]));
    }
    return std::move(rotations);
}

std::map<std::string, std::vector<std::tuple<Eigen::Vector3d, double>>> GetFingers(
    const std::vector<Eigen::Vector3d>& landmark_list, const std::map<std::string, std::vector<int>>& fingerIndicesMap) {
    std::map<std::string, std::vector<std::tuple<Eigen::Vector3d, double>>> fingerNameToRotations;

    // Normalize Vector by hand pose.
    Eigen::Matrix3d baseMatrix;
    baseMatrix << 1, 0, 0,
                  0, 1, 0,
                  0, 0, 1;
    auto rotation = EstimatePalmAngleFromBase(landmark_list, baseMatrix);
    std::vector<Eigen::Vector3d> directionNormalizedLandmarks;
    for (const auto& point : landmark_list) {
        directionNormalizedLandmarks.push_back(rotation.inverse() * point);
    }

    // Get the rotations for each fingers.
    for (const auto& tuple : fingerIndicesMap) {
        fingerNameToRotations[std::get<0>(tuple)] = GetRelativeAnglesFromXYPlane(
            directionNormalizedLandmarks, std::get<1>(tuple)
        );
    }

    return fingerNameToRotations;
}

PYBIND11_MODULE(landmark_utils, m) {
    // m.def("get_shortest_rotvec_between_two_vector", &GetRotationVector);
    m.def("get_shortest_rotvec_between_two_vector", &GetRotationVector, py::return_value_policy::move);
    m.def("get_fingers", &GetFingers, py::return_value_policy::move);
    m.def("estimate_palm_rotation", &EstimatePalmRotation, py::return_value_policy::move);
}