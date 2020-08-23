#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

class YesOrNoRecognizer {
 private:
  /* data */
 public:
  enum SignType { Yes = 0, No, Nothing };

  YesOrNoRecognizer(/* args */);
  ~YesOrNoRecognizer();

  GetCurrentState() {}
};

YesOrNoRecognizer::YesOrNoRecognizer(/* args */) {}

YesOrNoRecognizer::~YesOrNoRecognizer() {}

PYBIND11_MODULE(yes_or_no, m) {
  py::class_<GraphRunner>(m, "YesOrNoRecognizer") yes_or_no_recognizer;

  yes_or_no_recognizer.def(py::init<>())
      .def("get_current_state", &GraphRunner::GetCurrentState);

  py::enum_<YesOrNoRecognizer::SignType>(yes_or_no_recognizer, "SignType")
      .value("Yes", YesOrNoRecognizer::SignType::Yes)
      .value("No", YesOrNoRecognizer::SignType::No)
      .value("Nothing", YesOrNoRecognizer::SignType::Nothing)
      .export_values();
}