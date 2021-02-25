#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/object_detection/anchor.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

TEST(TensorRemapCalculatorTest, BasicTest) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "TensorRemapCalculator"
    input_stream: "TENSORS:detection_tensors"
    output_stream: "TENSORS:fixed_detection_tensors"
  )"));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
}
}  // namespace mediapipe