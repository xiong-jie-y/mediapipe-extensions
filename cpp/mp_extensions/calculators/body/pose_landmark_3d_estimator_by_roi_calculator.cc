//  BY Nicholas, nickadamu@gmail.com
// This a custom gesture recognizer

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include <cmath>
#include <vector>

namespace mediapipe
{
namespace
{
constexpr char normGpuImage[] = "IMAGE_GPU";
constexpr char normalizedLandmarkListTag[] = "LANDMARKS";

} // namespace

class PoseLandmark3DEstimatorByRoiCalculator : public CalculatorBase
{
public:
    PoseLandmark3DEstimatorByRoiCalculator(){};

    static ::mediapipe::Status GetContract(CalculatorContract *cc);

    ::mediapipe::Status Open(CalculatorContext *cc) override;

    ::mediapipe::Status Process(CalculatorContext *cc) override;

private:
};
REGISTER_CALCULATOR(PoseLandmark3DEstimatorByRoiCalculator);

// Verifies input and output packet types
::mediapipe::Status GestureRecognizerCalculator::GetContract(CalculatorContract *cc)
{
    RET_CHECK(cc->Inputs().HasTag(normGpuImage));
    cc->Inputs().Tag(normGpuImage).Set<std::vector<NormalizedLandmarkList>>();

    // Checks if input stream has the normRectTag
    RET_CHECK(cc->Inputs().HasTag(normRectTag));
    // Set normRectTag to receive a NormalizedRect as input
    cc->Inputs().Tag(normRectTag).Set<std::vector<NormalizedRect>>();

    // Check if output stream has tag recognizedHandGestureTag
    RET_CHECK(cc->Outputs().HasTag(recognizedHandGestureTag));
    // Set output stream to recognizedHandGesture string
    cc->Outputs().Tag(recognizedHandGestureTag).Set<std::vector<std::string>>();

    return ::mediapipe::OkStatus();
}

::mediapipe::Status GestureRecognizerCalculator::Open(CalculatorContext *cc)
{
    // Must look into this carefully
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
}

std::string GestureRecognizerCalculator::GetSignType(
    const NormalizedLandmarkList& landmarkList, const NormalizedRect& rect) {
    std::string *gesture_text;

    float width = rect.width();
    float height = rect.height();

    if (width < 0.01 || height < 0.01)
    {
        gesture_text = new std::string("Finding hand");
        return *gesture_text;
    }

    // finger states
    bool thumbIsOpen = false;
    bool firstFingerIsOpen = false;
    bool secondFingerIsOpen = false;
    bool thirdFingerIsOpen = false;
    bool fourthFingerIsOpen = false;

    float pseudoFixKeyPoint = landmarkList.landmark(2).x();
    if (landmarkList.landmark(3).x() < pseudoFixKeyPoint && landmarkList.landmark(4).x() < pseudoFixKeyPoint)
    {
        thumbIsOpen = true;
    }

    pseudoFixKeyPoint = landmarkList.landmark(6).y();
    if (landmarkList.landmark(7).y() < pseudoFixKeyPoint && landmarkList.landmark(8).y() < pseudoFixKeyPoint)
    {
        firstFingerIsOpen = true;
    }

    pseudoFixKeyPoint = landmarkList.landmark(10).y();
    if (landmarkList.landmark(11).y() < pseudoFixKeyPoint && landmarkList.landmark(12).y() < pseudoFixKeyPoint)
    {
        secondFingerIsOpen = true;
    }

    pseudoFixKeyPoint = landmarkList.landmark(14).y();
    if (landmarkList.landmark(15).y() < pseudoFixKeyPoint && landmarkList.landmark(16).y() < pseudoFixKeyPoint)
    {
        thirdFingerIsOpen = true;
    }
 
    pseudoFixKeyPoint = landmarkList.landmark(18).y();
    if (landmarkList.landmark(19).y() < pseudoFixKeyPoint && landmarkList.landmark(20).y() < pseudoFixKeyPoint)
    {
        fourthFingerIsOpen = true;
    }

    // Hand gesture recognition
    if (thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && thirdFingerIsOpen && fourthFingerIsOpen)
    {
        LOG(INFO) << "FIVE!";
        gesture_text = new std::string("FIVE");
    }
    else if (!thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && thirdFingerIsOpen && fourthFingerIsOpen)
    {
        LOG(INFO) << "FOUR!";
        gesture_text = new std::string("FOUR");
    }
    else if (thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        LOG(INFO) << "THREE!";
        gesture_text = new std::string("THREE");
    }
    else if (thumbIsOpen && firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        LOG(INFO) << "TWO!";
        gesture_text = new std::string("TWO");
    }
    else if (!thumbIsOpen && firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        LOG(INFO) << "ONE!";
        gesture_text = new std::string("ONE");
    }
    else if (!thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        LOG(INFO) << "TWO SURE!";
        gesture_text = new std::string("TWO SURE");
    }
    else if (!thumbIsOpen && firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && fourthFingerIsOpen)
    {
        LOG(INFO) << "ROCK!";
        gesture_text = new std::string("ROCK");
    }
    else if (thumbIsOpen && firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && fourthFingerIsOpen)
    {
        LOG(INFO) << "SPIDERMAN!";
        gesture_text = new std::string("SPIDERMAN");
    }
    else if (!thumbIsOpen && !firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        LOG(INFO) << "FIST!";
        gesture_text = new std::string("FIST");
    }
    else if (!firstFingerIsOpen && secondFingerIsOpen && thirdFingerIsOpen && fourthFingerIsOpen && this->isThumbNearFirstFinger(landmarkList.landmark(4), landmarkList.landmark(8)))
    {
        LOG(INFO) << "OK!";
        gesture_text = new std::string("OK");
    }
    else
    {
        LOG(INFO) << "Finger States: " << thumbIsOpen << firstFingerIsOpen << secondFingerIsOpen << thirdFingerIsOpen << fourthFingerIsOpen;
        LOG(INFO) << "___";
        gesture_text = new std::string("____");
    }
    return *gesture_text;
}

::mediapipe::Status GestureRecognizerCalculator::Process(CalculatorContext *cc)
{
    const auto &landmarks =
        cc->Inputs().Tag("LANDMARKS").Get<std::vector<NormalizedLandmarkList>>();
    const auto &norm_rects =
        cc->Inputs().Tag("NORM_RECTS").Get<std::vector<NormalizedRect>>();

    assert(landmarks.size() == norm_rects.size());

    std::vector<std::string>* gesture_texts = new std::vector<std::string>();
    for (size_t i = 0; i < landmarks.size(); i++) {
        gesture_texts->push_back(GetSignType(landmarks[i], norm_rects[i]));
    }

    // We set output stream to recognized hand gesture text
    cc->Outputs().Tag(recognizedHandGestureTag).Add(gesture_texts, cc->InputTimestamp());

    return ::mediapipe::OkStatus();
}

} // namespace mediapipe