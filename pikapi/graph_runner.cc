#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>


#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include "mediapipe/framework/port/map_util.h"

#include "mediapipe/framework/formats/landmark.pb.h"

namespace py = pybind11;

constexpr char kInputStream[] = "input_video";
constexpr char kInputStream2[] = "input_depth_video";
constexpr char kOutputStream[] = "output_video";

class GraphRunner
{
public:
    GraphRunner(
        const std::string &graph_path, const std::vector<std::string> &channels,
        const pybind11::dict& input_side_packets)
    {
        std::string calculator_graph_config_contents;
        mediapipe::file::GetContents(
            graph_path, &calculator_graph_config_contents);
        LOG(INFO) << "Get calculator graph config contents: "
                  << calculator_graph_config_contents;
        config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
            calculator_graph_config_contents);

        LOG(INFO) << "Initialize the calculator graph.";
        auto status = graph.Initialize(config);
        if (!status.ok())
        {
            LOG(ERROR) << status;
        }

        LOG(INFO) << "Initialize the GPU.";
        auto maybe_gpu_res = std::move(mediapipe::GpuResources::Create());
        if (maybe_gpu_res.ok())
        {
            LOG(INFO) << "Succeeded get GPU";
            graph.SetGpuResources(std::move(maybe_gpu_res.ValueOrDie()));
            gpu_helper.InitializeForTest(graph.GetGpuResources().get());
        }
        else
        {
            LOG(ERROR) << maybe_gpu_res.status();
        }
        // std::shared_ptr<mediapipe::GpuResources> gpu_resources = mediapipe::GpuResources::Create().ValueOrDie();
        // auto status =
        // LOG(INFO) << status;
        // gpu_helper.InitializeForTest(graph.GetGpuResources().get());

        LOG(INFO) << "Start running the calculator graph.";
        // ASSIGN_OR_RETURN(*poller.get(),
        //                 graph.AddOutputStreamPoller(kOutputStream));
        {
            mediapipe::StatusOrPoller maybe_poller = std::move(graph.AddOutputStreamPoller(kOutputStream));
            if (maybe_poller.ok())
            {
                poller = std::make_shared<mediapipe::OutputStreamPoller>(
                    std::move(maybe_poller).ValueOrDie());
            }
            else
            {
                LOG(ERROR) << maybe_poller.status();
            }
        }
        for (const auto &a : channels)
        {
            mediapipe::StatusOrPoller maybe_poller = std::move(graph.AddOutputStreamPoller(a));
            if (maybe_poller.ok())
            {
                channel_name_to_poller_[a] = std::make_shared<mediapipe::OutputStreamPoller>(
                    std::move(maybe_poller).ValueOrDie());
            }
            else
            {
                LOG(ERROR) << maybe_poller.status();
            }
        }

        std::map<std::string, mediapipe::Packet> input_side_packet_map;
        for (const auto& kv_pair : input_side_packets) {
          InsertIfNotPresent(&input_side_packet_map,
                             kv_pair.first.cast<std::string>(),
                             kv_pair.second.cast<mediapipe::Packet>());
        }

        auto start_run = graph.StartRun(input_side_packet_map);
        if (!start_run.ok())
            LOG(ERROR) << start_run;
    }
    ~GraphRunner()
    {
        graph.CloseInputStream(kInputStream);
        graph.WaitUntilDone();
    }

    std::unique_ptr<mediapipe::ImageFrame> CreateImageFrameFromNumpyByCopy(py::array_t<unsigned char> &input)
    {
        LOG(INFO) << "Create image";
        py::buffer_info buf = input.request();
        cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char *)buf.ptr);

        // Wrap Mat into an ImageFrame.
        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, mat.cols, mat.rows,
            mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        mat.copyTo(input_frame_mat);
        // LOG(INFO) << "aho";
        // auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        //     mediapipe::ImageFormat::SRGB, buf.shape[0], buf.shape[1],
        //     (buf.shape[0] * buf.shape[2]), (uint8*)buf.ptr);
        return std::move(input_frame);
    }

    // mediapipe::Status AddInputFrameAsGpuBufferPacket(mediapipe::CalculatorGraph& Graph, char inputStreamName[], mediapipe::ImageFrame& input_frame) {
    //         // Convert ImageFrame to GpuBuffer.
    //         auto texture = gpu_helper.CreateSourceTexture(*input_frame.get());
    //         auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
    //         glFlush();
    //         texture.Release();

    //         // Send GPU image packet into the graph.
    //         graph.AddPacketToInputStream(
    //             inputStreamName, mediapipe::Adopt(gpu_frame.release())
    //                               .At(mediapipe::Timestamp(frame_timestamp_us)));

    //         return ::mediapipe::OkStatus();
    // }

    // py::array_t<unsigned char> ProcessRGBDFrame(py::array_t<unsigned char> &rgb_image, py::array_t<unsigned char> &depth_image) {
    //     LOG(INFO) << "Creating RGB Image";
    //     auto rgb_frame = CreateImageFrameFromNumpyByCopy(rgb_image);

    //     LOG(INFO) << "Creating Depth Image";
    //     // Depth Image
    //     py::buffer_info buf = input.request();
    //     cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC1, (unsigned char *)buf.ptr);

    //     // Wrap Mat into an ImageFrame.
    //     auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
    //         mediapipe::ImageFormat::SRGB, mat.cols, mat.rows,
    //         mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
    //     cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    //     mat.copyTo(input_frame_mat);

    //     LOG(INFO) << "Run Graph";
    //     gpu_helper.RunInGlContext([this, &rgb_frame, &input_frame, &frame_timestamp_us]() -> ::mediapipe::Status {
    //         MP_RETURN_IF_ERROR(AddInputFrameAsGpuBufferPacket(graph, kInputStream, rgb_frame));
    //         MP_RETURN_IF_ERROR(AddInputFrameAsGpuBufferPacket(graph, kInputStream2, input_frame));
    //     });
    // }

    py::array_t<unsigned char> ProcessFrame(py::array_t<unsigned char> &input)
    {
        if (input.ndim() != 3)
            throw std::runtime_error("1-channel image must be 2 dims ");

        auto input_frame = CreateImageFrameFromNumpyByCopy(input);

        // Prepare and add graph input packet.
        size_t frame_timestamp_us =
            (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;

        
        LOG(INFO) << "RunGlContext";
        auto run_graph_status = gpu_helper.RunInGlContext([this, &input_frame, &frame_timestamp_us]() -> ::mediapipe::Status {
            // Convert ImageFrame to GpuBuffer.
            auto texture = gpu_helper.CreateSourceTexture(*input_frame.get());
            auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
            glFlush();
            texture.Release();
            // Send GPU image packet into the graph.
            MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
                kInputStream, mediapipe::Adopt(gpu_frame.release())
                                  .At(mediapipe::Timestamp(frame_timestamp_us))));

            return ::mediapipe::OkStatus();
        });
        if (!run_graph_status.ok()) {
            LOG(INFO) << run_graph_status;
        }

        // Get the graph result packet, or stop if that fails.
        mediapipe::Packet packet;
        if (!poller->Next(&packet)) {
            LOG(INFO) << "error getting packet";
        }

        // break;
        std::unique_ptr<mediapipe::ImageFrame> output_frame;

        LOG(INFO) << "RunFetch";
        // Convert GpuBuffer to ImageFrame.
        auto run_fetch_status = gpu_helper.RunInGlContext(
            [this, &packet, &output_frame]() -> ::mediapipe::Status {
                auto &gpu_frame = packet.Get<mediapipe::GpuBuffer>();
                auto texture = gpu_helper.CreateSourceTexture(gpu_frame);
                output_frame = absl::make_unique<mediapipe::ImageFrame>(
                    mediapipe::ImageFormatForGpuBufferFormat(gpu_frame.format()),
                    gpu_frame.width(), gpu_frame.height(),
                    mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
                gpu_helper.BindFramebuffer(texture);
                const auto info =
                    mediapipe::GlTextureInfoForGpuBufferFormat(gpu_frame.format(), 0);
                glReadPixels(0, 0, texture.width(), texture.height(), info.gl_format,
                             info.gl_type, output_frame->MutablePixelData());
                glFlush();
                texture.Release();
                return ::mediapipe::OkStatus();
            });
        if (!run_fetch_status.ok()) {
            LOG(INFO) << run_fetch_status;
        }
        LOG(INFO) << "Convert Image";

        // Convert back to opencv for display or saving.
        cv::Mat output_frame_mat = mediapipe::formats::MatView(output_frame.get());

        int size = output_frame_mat.channels() * output_frame_mat.size().width * output_frame_mat.size().height;
        py::array_t<unsigned char> result(size);
        py::buffer_info buf2 = result.request();

        auto pt = output_frame_mat.data;
        unsigned char *dstPt = (unsigned char *)buf2.ptr;

        for (int i = 0; i < size; i++)
        {
            dstPt[i] = pt[i];
        }

        LOG(INFO) << "Getting other outputs";
        // Get other channels.
        for (const auto &[key, poller] : channel_name_to_poller_)
        {
            // LOG(INFO) << poller->QueueSize();
            if (poller->QueueSize() == 0) {
                channel_name_to_pocket_[key] = nullptr;
                continue;
            } else {
                mediapipe::Packet packet;
                if (!poller->Next(&packet))
                    LOG(INFO) << "error getting packet";
                LOG(INFO) << "Getting " << key;
                channel_name_to_pocket_[key] = std::make_shared<mediapipe::Packet>(packet);
            }
        }

        return result;
    }
    // template <typename T>
    // const T &GetLatestObject(std::string name)
    // {
    //     return channel_name_to_pocket_[name].Get<T>();
    // }

    template <typename T>
    std::vector<std::vector<Eigen::Vector3d>> GetPoint3DListsFromLandmark(std::string name)
    {
        if (channel_name_to_pocket_.find(name) == channel_name_to_pocket_.end()) {
            std::vector<std::vector<Eigen::Vector3d>> a(0);
            return a;
        }

        if (!channel_name_to_pocket_[name]) {
            std::vector<std::vector<Eigen::Vector3d>> a(0);
            return a;
        }

        const auto &landmark_lists = channel_name_to_pocket_[name]->Get<T>();

        if (landmark_lists.size() == 0)
        {
            std::vector<std::vector<Eigen::Vector3d>> a(0);
            return a;
        }

        auto dim_1 = landmark_lists.size();
        auto dim_2 = landmark_lists[0].landmark().size();

        std::vector<std::vector<Eigen::Vector3d>> point_3d_lists(dim_1,
                                                                 std::vector<Eigen::Vector3d>(dim_2));
        for (size_t i = 0; i < dim_1; i++)
        {
            for (size_t j = 0; j < dim_2; j++)
            {
                const auto &landmark_point = landmark_lists[i].landmark().at(j);
                point_3d_lists[i][j] = Eigen::Vector3d(landmark_point.x(), landmark_point.y(), landmark_point.z());
            }
        }

        return point_3d_lists;
    }

    template <typename T>
    std::vector<Eigen::Vector3d> GetPoint3DListFromLandmark(std::string name)
    {
        if (channel_name_to_pocket_.find(name) == channel_name_to_pocket_.end()) {
            std::vector<Eigen::Vector3d> a(0);
            return a;
        }

        if (!channel_name_to_pocket_[name]) {
            std::vector<Eigen::Vector3d> a(0);
            return a;
        }

        const auto &landmark_list = channel_name_to_pocket_[name]->Get<T>();

        auto dim_2 = landmark_list.landmark().size();

        std::vector<Eigen::Vector3d> point_3d_lists(dim_2);

        for (size_t j = 0; j < dim_2; j++)
        {
            const auto &landmark_point = landmark_list.landmark().at(j);
            point_3d_lists[j] = Eigen::Vector3d(landmark_point.x(), landmark_point.y(), landmark_point.z());
        }

        return point_3d_lists;
    }


    py::object GetFloat(std::string name)
    {
        if (channel_name_to_pocket_.find(name) == channel_name_to_pocket_.end()) {
            return py::cast<py::none>(Py_None);
        }

        if (!channel_name_to_pocket_[name]) {
            return py::cast<py::none>(Py_None);
        }

        return py::cast(channel_name_to_pocket_[name]->Get<float>());
    }

    // template <typename T>
    // std::string GetProtobufObject(std::string name)
    // {
    //     std::string serialized_str;
    //     channel_name_to_pocket_[name].Get<T>().SerializeToString(&serialized_str);
    //     return serialized_str;
    // }

private:
    mediapipe::CalculatorGraphConfig config;
    mediapipe::CalculatorGraph graph;

    mediapipe::GlCalculatorHelper gpu_helper;

    std::shared_ptr<mediapipe::OutputStreamPoller> poller;

    std::map<std::string, std::shared_ptr<mediapipe::OutputStreamPoller>> channel_name_to_poller_;
    std::map<std::string, std::shared_ptr<mediapipe::Packet>> channel_name_to_pocket_;
};

PYBIND11_MODULE(graph_runner, m)
{
    py::class_<GraphRunner>(m, "GraphRunner")
        .def(py::init<const std::string &, const std::vector<std::string> &, const pybind11::dict&>())
        .def("process_frame", &GraphRunner::ProcessFrame)
        .def("get_normalized_landmark_lists", &GraphRunner::GetPoint3DListsFromLandmark<std::vector<mediapipe::NormalizedLandmarkList>>)
        .def("get_normalized_landmark_list", &GraphRunner::GetPoint3DListFromLandmark<mediapipe::NormalizedLandmarkList>)
        .def("get_float", &GraphRunner::GetFloat)
        .def("get_landmark_lists", &GraphRunner::GetPoint3DListsFromLandmark<std::vector<mediapipe::LandmarkList>>);
    // .def("get_landmark_lists_pb", &GraphRunner::GetProtobufObject<std::vector<mediapipe::NormalizedLandmarkList>>);
    // .def("process_rgbd_frame", &GraphRunner::ProcessRGBDFrame);

    // py::class_<mediapipe::NormalizedLandmark>(m, "NormalizedLandmark")
    //     .def_readwrite("landmark", &mediapipe::NormalizedLandmark::x, "``float64`` array of shape ``(3, )``");

    // py::class_<mediapipe::NormalizedLandmarkList>(m, "NormalizedLandmarkList")
    //     .def_readwrite("landmark", &mediapipe::NormalizedLandmarkList::landmark, "``float64`` array of shape ``(3, )``");
}