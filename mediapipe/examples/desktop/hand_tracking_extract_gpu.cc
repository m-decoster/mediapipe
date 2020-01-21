// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
// This example requires a linux computer and a GPU with EGL support drivers.

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <glob.h> // Only tested on Ubuntu!

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "outputs";

DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");
DEFINE_string(input_video_path, "",
              "Full path of video to load. "
              "If not provided, looks at input_image_dir_path.");
DEFINE_string(input_image_dir_path, "",
              "Full path of directory of images to load. "
              "If not provided, fails.");
DEFINE_string(output_file_path, "",
              "Full path of where to save result (.txt only). ");

// based on https://stackoverflow.com/questions/8401777/simple-glob-in-c-on-unix-system/8615450#8615450
std::vector<std::string> glob(const std::string &pattern) {
  glob_t glob_result = {0};
  int ret = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
  if (ret) {
    globfree(&glob_result);
    std::stringstream ss;
    ss << "glob() failed with return_value " << ret << std::endl;
    throw std::runtime_error(ss.str());
  }
  std::vector<std::string> result;
  result.reserve(glob_result.gl_pathc);
  for(size_t i = 0; i < glob_result.gl_pathc; ++i) {
    result.push_back(std::string(glob_result.gl_pathv[i]));
  }
  globfree(&glob_result);
  return result;
}

::mediapipe::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      FLAGS_calculator_graph_config_file, &calculator_graph_config_contents));
  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Initialize the GPU.";
  ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
  MP_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));
  mediapipe::GlCalculatorHelper gpu_helper;
  gpu_helper.InitializeForTest(graph.GetGpuResources().get());

  LOG(INFO) << "Load the video or list files in the directory.";
  cv::VideoCapture capture;
  std::vector<std::string> image_files;
  const bool load_video = !FLAGS_input_video_path.empty();
  if (load_video) {
    capture.open(FLAGS_input_video_path);
    RET_CHECK(capture.isOpened());
  } else {
    std::string suffix = FLAGS_input_image_dir_path.back() == '/' ? "*.png" : "/*.png";
    image_files = glob(FLAGS_input_image_dir_path + suffix);
  }

  LOG(INFO) << "Start running the calculator graph.";
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                   graph.AddOutputStreamPoller(kOutputStream));
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "Start grabbing and processing frames.";
  size_t frame_timestamp = 0;
  bool grab_frames = true;
  std::ofstream ofs; // Output file stream for the floats
  ofs.open(FLAGS_output_file_path);
  int frame_counter = 0;
  while (grab_frames) {
    cv::Mat camera_frame;
    if (load_video) {
      // Capture opencv camera or video frame.
      cv::Mat camera_frame_raw;
      capture >> camera_frame_raw;
      if (camera_frame_raw.empty()) break;  // End of video.
      cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    } else {
      if (frame_counter >= image_files.size()) break; // End of dir.
      std::string filename = image_files.at(frame_counter);
      cv::Mat camera_frame_raw = cv::imread(filename);
      if (camera_frame_raw.empty()) break;  // For some reason we couldn't read this file.
      cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB); 
    }

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Prepare and add graph input packet.
    MP_RETURN_IF_ERROR(
        gpu_helper.RunInGlContext([&input_frame, &frame_timestamp, &graph,
                                   &gpu_helper]() -> ::mediapipe::Status {
          // Convert ImageFrame to GpuBuffer.
          auto texture = gpu_helper.CreateSourceTexture(*input_frame.get());
          auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
          glFlush();
          texture.Release();
          // Send GPU image packet into the graph.
          MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
              kInputStream, mediapipe::Adopt(gpu_frame.release())
                                .At(mediapipe::Timestamp(frame_timestamp++))));
          return ::mediapipe::OkStatus();
        }));

    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    if (!poller.Next(&packet)) break;
    std::vector<float> output_floats = packet.Get<std::vector<float>>();

    // Write to the output stream
    // One line per frame
    for (int fi = 0; fi < output_floats.size(); ++fi) {
        ofs << output_floats[fi];
        if (fi != output_floats.size() - 1) { // Skip last comma
            ofs << ",";
        }
    }
    ofs << std::endl;
    
    ++frame_counter;
  }

  LOG(INFO) << "Shutting down.";
  ofs.close();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::mediapipe::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
  } else {
    LOG(INFO) << "Success!";
  }
  return 0;
}
