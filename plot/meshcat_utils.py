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

# Crocoddyl tools
from crocoddyl.libcrocoddyl_pywrap import *  # noqa

cwd = os.getcwd()
sys.path.append(cwd)


def get_force_trajectory_from_solver(solver):
    """
    Snippet copied from Crocoddyl's DisplayAbstract class
    """
    fs = []
    models = [*solver.problem.runningModels.tolist(), solver.problem.terminalModel]
    datas = [*solver.problem.runningDatas.tolist(), solver.problem.terminalData]
    for i, data in enumerate(datas):
        model = models[i]
        if hasattr(data, "differential"):
            if isinstance(
                data.differential,
                DifferentialActionDataContactFwdDynamics,
            ) or isinstance(
                data.differential,
                DifferentialActionDataContactInvDynamics,
            ):
                fc = []
                for (
                    key,
                    contact,
                ) in data.differential.multibody.contacts.contacts.todict().items():
                    if model.differential.contacts.contacts[key].active:
                        joint = model.differential.state.pinocchio.frames[
                            contact.frame
                        ].parent
                        oMf = contact.pinocchio.oMi[joint] * contact.jMf
                        fiMo = pin.SE3(
                            contact.pinocchio.oMi[joint].rotation.T,
                            contact.jMf.translation,
                        )
                        force = fiMo.actInv(contact.f)
                        R = np.eye(3)
                        mu = 0.7
                        for k, c in model.differential.costs.costs.todict().items():
                            if isinstance(
                                c.cost.residual,
                                ResidualModelContactFrictionCone,
                            ):
                                if contact.frame == c.cost.residual.id:
                                    R = c.cost.residual.reference.R
                                    mu = c.cost.residual.reference.mu
                                    continue
                        fc.append(
                            {
                                "key": str(joint),
                                "oMf": oMf,
                                "f": force,
                                "R": R,
                                "mu": mu,
                            }
                        )
                fs.append(fc)
            elif isinstance(data.differential, StdVec_DiffActionData):
                fc = []
                for key, contact in (
                    data.differential[0]
                    .multibody.contacts.contacts.todict()
                    .items()
                ):
                    if model.differential.contacts.contacts[key].active:
                        joint = model.differential.state.pinocchio.frames[
                            contact.frame
                        ].parent
                        oMf = contact.pinocchio.oMi[joint] * contact.jMf
                        fiMo = pin.SE3(
                            contact.pinocchio.oMi[joint].rotation.T,
                            contact.jMf.translation,
                        )
                        force = fiMo.actInv(contact.fext)
                        R = np.eye(3)
                        mu = 0.7
                        for k, c in model.differential.costs.costs.todict().items():
                            if isinstance(
                                c.cost.residual,
                                ResidualModelContactFrictionCone,
                            ):
                                if contact.frame == c.cost.residual.id:
                                    R = c.cost.residual.reference.R
                                    mu = c.cost.residual.reference.mu
                                    continue
                        fc.append(
                            {
                                "key": str(joint),
                                "oMf": oMf,
                                "f": contact.fext,
                                "R": R,
                                "mu": mu,
                            }
                        )
                fs.append(fc)
        elif isinstance(data, ActionDataImpulseFwdDynamics):
            fc = []
            for key, impulse in data.multibody.impulses.impulses.todict().items():
                if model.impulses.impulses[key].active:
                    joint = model.state.pinocchio.frames[impulse.frame].parent
                    oMf = impulse.pinocchio.oMi[joint] * impulse.jMf
                    fiMo = pin.SE3(
                        impulse.pinocchio.oMi[joint].rotation.T,
                        impulse.jMf.translation,
                    )
                    force = fiMo.actInv(impulse.f)
                    R = np.eye(3)
                    mu = 0.7
                    for k, c in model.costs.costs.todict().items():
                        if isinstance(
                            c.cost.residual,
                            ResidualModelContactFrictionCone,
                        ):
                            if impulse.frame == c.cost.residual.id:
                                R = c.cost.residual.reference.R
                                mu = c.cost.residual.reference.mu
                                continue
                    fc.append(
                        {
                            "key": str(joint),
                            "oMf": oMf,
                            "f": force,
                            "R": R,
                            "mu": mu,
                        }
                    )
            fs.append(fc)
    return fs


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

    def add_robot(self, robot_name, pin_rob_model, collision_model, visual_model,
                  rob_position, rob_quaternion):
        viz = MeshcatVisualizer(pin_rob_model, collision_model, visual_model)
        viz.initViewer(self.viz.viewer)
        viz.loadViewerModel(rootNodeName=robot_name)

        tf_transl = tf.translation_matrix(rob_position)
        tf_rot = tf.quaternion_matrix(rob_quaternion)
        tf_pose = tf.concatenate_matrices(tf_transl, tf_rot)
        viz.viewer[robot_name].set_transform(tf_pose)

    def add_arrow(self, obj_name, color=[1, 0, 0], height=0.1):
        arrow_shaft = g.Cylinder(height, 0.01)
        arrow_head = g.Cylinder(0.04, 0.04, radiusTop=0.001, radiusBottom=0.04)
        material = g.MeshPhongMaterial()
        material.color = int(color[0] * 255) * 256 ** 2 + int(
            color[1] * 255) * 256 + int(color[2] * 255)

        arrow_offset = tf.translation_matrix([0., height/2., 0.])
        shaft_rotation = tf.rotation_matrix(np.pi/2., [1., 0., 0.])
        # arrow_vertical = tf.concatenate_matrices(arrow_offset, shaft_rotation)
        self.viz.viewer[obj_name]["arrow"].set_object(arrow_shaft, material)
        self.viz.viewer[obj_name]["arrow"].set_transform(shaft_rotation)
        self.viz.viewer[obj_name]["arrow/head"].set_object(arrow_head, material)
        self.viz.viewer[obj_name]["arrow/head"].set_transform(arrow_offset)

    def displayForcesFromCrocoddylSolver(self, fs_ti, frame):
        # fs = get_force_trajectory_from_solver(solver)
        # for ti in range(len(fs)):
            for contact in range(len(fs_ti)):
                pos = fs_ti[contact]['oMf'].translation
                force_ori = fs_ti[contact]['oMf'].rotation
                force_dir = fs_ti[contact]['f'].linear

                scale = force_dir[2] / 1000.
                # tf_S = tf.scale_matrix(scale, [0., 0., 0.], [0., 0., 1.])
                tf_S = tf.translation_matrix([0., 0., scale])
                tf_pos = tf.translation_matrix(pos)
                tf_pos[:3, :3] = force_ori
                T_force = tf.concatenate_matrices(tf_pos, tf_S)
                link_name = self.model.names[int(fs_ti[contact]['key'])]
                frame['forces'][link_name].set_transform(T_force)
                # self.viz.viewer['forces'][link_name]["arrow"].set_transform(
                #     fs[ti][contact]['oMf'].homogeneous)

    def displayFromCrocoddylSolver(self, solver):
        for it in solver:
            models = it.problem.runningModels.tolist() + [it.problem.terminalModel]
            dts = [m.dt if hasattr(m, "differential") else 0. for m in models]

            fs = get_force_trajectory_from_solver(it)

            for sim_time_idx in np.arange(0, len(dts), self.save_freq):
                q = np.array(it.xs[int(sim_time_idx)][:self.robot_nq])
                self.viz.display(q)

                fs_ti = fs[sim_time_idx]

                with self.anim.at_frame(self.viz.viewer, self.frame_idx) as frame:
                    self.display_visualizer_frames(frame, q)
                    self.displayForcesFromCrocoddylSolver(fs_ti, frame)

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
            self.viz.viewer[end_effector_name + "/" + str(i)].set_object(g.Sphere(0.01),  material)
            Href = np.array(
                [
                    [1.0, 0.0, 0.0, target[0]],
                    [0.0, 1.0, 0.0, target[1]],
                    [0.0, 0.0, 1.0, target[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            self.viz.viewer[end_effector_name+"/" + str(i)].set_transform(Href)


    def hide_visuals(self, viz_list):
        for viz in viz_list:
            self.viz.viewer[viz].set_property("visible", False)
