import open3d as o3d


def create_lineset(points, color):
    lines = [
        [0, 1],
    ]
    colors = [color]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


class PoseVisualizer():
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=[0, 0, 0])
        self.vis.add_geometry(mesh_frame)
        self._geom_map = {}
        self._color_map = [
            [1,0,0], [0,1,0]
        ]
        print("Color map")
        for i, color in enumerate(self._color_map):
            print(f"Color of {i}: {color}")

    def update(self, geom_id, direction):
        if geom_id not in self._geom_map:
            color = self._color_map[geom_id]
            new_pose_line = create_lineset([[0, 0, 0], [1, 1, 1]], color)
            self.vis.add_geometry(new_pose_line)
            self._geom_map[geom_id] = new_pose_line

        pose_line = self._geom_map[geom_id]
        pose_line.points = o3d.utility.Vector3dVector([
            [0, 0, 0], direction
        ])
        self.vis.update_geometry(pose_line)
        self.vis.poll_events()
        self.vis.update_renderer()
