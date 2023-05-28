import os
import sys
import numpy as np
import pinocchio as pin

# Pinocchio Meshcat
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf
# Python-Meshcat
from meshcat.animation import Animation
from pinocchio.visualize.meshcat_visualizer import isMesh

cwd = os.getcwd()
sys.path.append(cwd)


class MeshcatPinocchioAnimation:
    def __init__(self, pin_robot_model, collision_model, visual_model,
                 robot_data, visual_data,
                 ctrl_freq=1000, save_freq=50):
        # self.robot = pin_robot_model
        self.robot_data = robot_data
        self.model = pin_robot_model
        self.robot_nq = pin_robot_model.nq
        self.viz = MeshcatVisualizer(self.model, collision_model, visual_model)
        try:
            self.viz.initViewer(open=True)
            self.viz.viewer.wait()
        except ImportError as err:
            print(
                "Error while initializing the viewer. It seems you should install Python meshcat"
            )
            print(err)
            sys.exit(0)
        self.viz.loadViewerModel(rootNodeName=self.model.name)

        # animation settings
        self.anim = Animation(default_framerate=ctrl_freq / save_freq)
        self.frame_idx = 0              # index of frame being saved in animation
        self.save_freq = save_freq      # display every save_freq simulation steps

        self.visual_model = visual_model
        self.visual_data = visual_data

    def displayFromCrocoddylSolver(self, solver):
        models = solver.problem.runningModels.tolist() + [solver.problem.terminalModel]
        dts = [m.dt if hasattr(m, "differential") else 0. for m in models]

        for sim_time_idx in np.arange(0, len(dts), self.save_freq):
            q = np.array(solver.xs[int(sim_time_idx)][:self.robot_nq])
            self.viz.display(q)

            with self.anim.at_frame(self.viz.viewer, self.frame_idx) as frame:
                self.display_visualizer_frames(frame, q)

            self.frame_idx += 1     # increase frame index counter

        # save animation
        self.viz.viewer.set_animation(self.anim, play=False)

    def display_visualizer_frames(self, frame, q):
        meshcat_visualizer = self.viz

        geom_model = self.visual_model
        geom_data = self.visual_data

        pin.forwardKinematics(self.model, self.robot_data, q)
        pin.updateGeometryPlacements(self.model, self.robot_data,
                                     geom_model, geom_data)
        for visual in geom_model.geometryObjects:
            viewer_name = meshcat_visualizer.getViewerNodeName(visual, pin.GeometryType.VISUAL)
            # Get mesh pose.
            M = geom_data.oMg[geom_model.getGeometryId(visual.name)]
            # Manage scaling
            if isMesh(visual):
                scale = np.asarray(visual.meshScale).flatten()
                S = np.diag(np.concatenate((scale, [1.0])))
                # S = visual.placement.homogeneous
                T = np.array(M.homogeneous).dot(S)
            else:
                T = M.homogeneous
            # Update viewer configuration.
            frame[viewer_name].set_transform(T)

    def display_targets(self, end_effector_name, targets, color=None):
        if color is None:
            color = [1, 0, 0]
        material = g.MeshPhongMaterial()
        material.color = int(color[0] * 255) * 256 ** 2 + int(
            color[1] * 255) * 256 + int(color[2] * 255)
        material.opacity = 0.4
        for i, target in enumerate(targets):
            self.viz.viewer[end_effector_name + "_" + str(i)].set_object(g.Sphere(0.01),  material)
            Href = np.array(
                [
                    [1.0, 0.0, 0.0, target[0]],
                    [0.0, 1.0, 0.0, target[1]],
                    [0.0, 0.0, 1.0, target[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            self.viz.viewer[end_effector_name + "_" + str(i)].set_transform(Href)

