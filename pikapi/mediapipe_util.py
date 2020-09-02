import mediapipe.python as mp

def create_packet_map(input_configs):
    packet_map = {}
    for key, value in input_configs.items():
        if isinstance(value, str):
            packet_map[key] = mp.packet_creator.create_string(value)
        elif isinstance(value, float):
            packet_map[key] = mp.packet_creator.create_float(value)
        else:
            raise RuntimeError(f"{type(value)} is not supported type for building protobuf.")
    return packet_map
