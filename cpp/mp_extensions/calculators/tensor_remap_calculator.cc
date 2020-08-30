#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "tensorflow/lite/interpreter.h"

namespace mediapipe {

constexpr char kTensorsTag[] = "TENSORS";

class TensorRemapCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    if (cc->Inputs().HasTag(kTensorsTag)) {
        cc->Inputs().Tag(kTensorsTag).Set<std::vector<TfLiteTensor>>();
    }
    if (cc->Outputs().HasTag(kTensorsTag)) {
        cc->Outputs().Tag(kTensorsTag).Set<std::vector<TfLiteTensor>>();
    }
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(0);
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    if (cc->Inputs().Tag(kTensorsTag).IsEmpty()) {
        return ::mediapipe::OkStatus();
    }
    const auto& input_tensors =
        cc->Inputs().Tag(kTensorsTag).Get<std::vector<TfLiteTensor>>();

    if (input_tensors.size() == 2) {
        auto output_tensors = absl::make_unique<std::vector<TfLiteTensor>>();
        output_tensors->emplace_back(input_tensors[1]);
        output_tensors->emplace_back(input_tensors[0]);
        cc->Outputs()
            .Tag(kTensorsTag)
            .Add(output_tensors.release(), cc->InputTimestamp());
    }
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Close(CalculatorContext* cc) override {
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(TensorRemapCalculator);

}  // namespace mediapipe
