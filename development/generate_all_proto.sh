set -e
protoc -I=formats/ --csharp_out=pikapi_sharp formats/perception_state.proto
protoc --proto_path=formats/ --python_out=pikapi/protos formats/perception_state.proto
python tools/generate_proto.py