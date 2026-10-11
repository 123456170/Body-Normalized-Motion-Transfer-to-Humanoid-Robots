from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("body_normalized_humanoid_demo")


# =============================================================================
# SAFETY AND SCOPE
# =============================================================================

HARDWARE_OUTPUT_ENV = "ROBOTICS_HARDWARE_OUTPUT"
SAFETY_REVIEW_PATH = Path("safety_review.json")


def hardware_output_requested() -> bool:
    return os.environ.get(HARDWARE_OUTPUT_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def enforce_simulation_only_scope() -> None:
    """
    Simulation-only guard.

    Hardware output is disabled by default. If explicitly requested through an
    environment variable, this app refuses to run unless a safety-review file
    exists. Even then, this demo does not command real hardware.
    """
    if hardware_output_requested():
        if not SAFETY_REVIEW_PATH.exists():
            raise RuntimeError(
                "Hardware output was requested, but safety_review.json is missing. "
                "This project is simulation-only unless an explicit safety review exists."
            )
        logger.warning(
            "Hardware-output flag detected, but this application still performs simulation only. "
            "No real actuation is implemented."
        )

    st.sidebar.markdown(
        """
        ### Safety scope
        - **Simulation only**
        - Hardware output disabled by default
        - No real robot is commanded
        - All motion data in this demo is synthetic/mock
        """
    )


# =============================================================================
# CONSTANTS AND TYPES
# =============================================================================

JOINT_NAMES = [
    "torso_pitch",
    "left_shoulder_pitch",
    "left_elbow",
    "right_shoulder_pitch",
    "right_elbow",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]
N_JOINTS = len(JOINT_NAMES)

REFERENCE_LEG_LENGTH = 0.90
REFERENCE_ARM_LENGTH = 0.65

LEG_FRACTIONS = {
    "thigh": 0.55,
    "shank": 0.45,
    "foot": 0.18,
    "torso": 0.48,
    "hip_width": 0.22,
}

ARM_FRACTIONS = {
    "upper": 0.55,
    "fore": 0.45,
    "shoulder_width": 0.34,
}

SKELETON_EDGES = [
    ("pelvis", "torso_top"),
    ("torso_top", "head"),
    ("torso_top", "left_shoulder"),
    ("torso_top", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("pelvis", "left_hip"),
    ("pelvis", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_toe"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_toe"),
]

ROBOT_COLORS = {
    "robot_A_train": "#2ca02c",
    "robot_B_train": "#17becf",
    "robot_C_heldout": "#d62728",
    "human_reference": "#7f7f7f",
}


@dataclass
class MorphologyDescriptor:
    robot_id: str
    leg_length: float
    arm_length: float
    total_mass: float
    joint_range_min: np.ndarray
    joint_range_max: np.ndarray
    torque_limit: np.ndarray
    joint_scale: float = 1.0

    def vector(self) -> np.ndarray:
        range_width = self.joint_range_max - self.joint_range_min
        return np.array(
            [
                self.leg_length,
                self.arm_length,
                self.total_mass / 100.0,
                self.joint_scale,
                float(np.mean(range_width)),
                float(np.mean(self.torque_limit)) / 100.0,
            ],
            dtype=float,
        )


@dataclass
class RobotModel:
    descriptor: MorphologyDescriptor
    calibration_offsets: np.ndarray
    plant_bias: np.ndarray
    license_note: str = "synthetic/mock"


@dataclass
class MotionClip:
    t: np.ndarray
    root: np.ndarray
    wrist_norm: np.ndarray
    ankle_norm: np.ndarray
    foot_contact: np.ndarray
    confidence: np.ndarray
    source: str = "synthetic_mock"

    @property
    def num_frames(self) -> int:
        return int(self.t.size)


@dataclass
class TrajectoryResult:
    robot_id: str
    q_desired: np.ndarray
    q_actual: np.ndarray
    root: np.ndarray
    points_frames: Optional[List[Dict[str, np.ndarray]]]
    ee_error_mm: np.ndarray
    foot_skate_mm_s: np.ndarray
    fall: np.ndarray
    joint_limit_hit: np.ndarray


@dataclass
class VariantResult:
    trajectories: Dict[str, TrajectoryResult]
    per_frame: pd.DataFrame
    summary: pd.DataFrame


@dataclass
class LinearMorphologyAdapter:
    coef: np.ndarray
    intercept: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    train_ids: List[str]
    fit_seconds: float

    @staticmethod
    def fit(models: Dict[str, RobotModel], train_ids: List[str], alpha: float = 1.0) -> "LinearMorphologyAdapter":
        t0 = time.perf_counter()

        X = np.vstack([models[rid].descriptor.vector() for rid in train_ids])
        Y = np.vstack([models[rid].calibration_offsets for rid in train_ids])

        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("Morphology inputs and calibration outputs must be 2D arrays.")

        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Input/output sample mismatch: X has {X.shape[0]} rows, "
                f"Y has {Y.shape[0]} rows."
            )

        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0) + 1e-6
        y_mean = Y.mean(axis=0)
        y_std = Y.std(axis=0) + 1e-6

        Xs = (X - x_mean) / x_std
        Ys = (Y - y_mean) / y_std

        Xa = np.column_stack([np.ones(Xs.shape[0]), Xs])

        A = Xa.T @ Xa + alpha * np.eye(Xa.shape[1])
        B = Xa.T @ Ys
        W = np.linalg.solve(A, B)

        if not np.all(np.isfinite(W)):
            raise RuntimeError("Morphology adapter fit produced non-finite weights.")

        fit_seconds = time.perf_counter() - t0

        return LinearMorphologyAdapter(
            coef=W[1:, :],
            intercept=W[0, :],
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            train_ids=list(train_ids),
            fit_seconds=fit_seconds,
        )

    def predict(self, descriptor: MorphologyDescriptor) -> np.ndarray:
        x = np.asarray(descriptor.vector(), dtype=float).reshape(1, -1)

        if x.shape[1] != self.x_mean.size:
            raise ValueError(
                f"Morphology descriptor has {x.shape[1]} features, "
                f"but adapter expects {self.x_mean.size}."
            )

        xs = (x - self.x_mean) / self.x_std
        xa = np.column_stack([np.ones(xs.shape[0]), xs])

        W = np.vstack([self.intercept.reshape(1, -1), self.coef])

        if xa.shape[1] != W.shape[0]:
            raise ValueError(
                f"Adapter design matrix has {xa.shape[1]} columns, "
                f"but weight matrix has {W.shape[0]} rows."
            )

        ys = xa @ W
        out = (ys[0] * self.y_std) + self.y_mean

        if not np.all(np.isfinite(out)):
            raise RuntimeError("Morphology adapter prediction produced non-finite values.")

        return np.asarray(out, dtype=float)


# =============================================================================
# ROBOT ZOO
# =============================================================================


def default_joint_ranges(scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    mins = np.array(
        [
            -0.50,  # torso pitch
            -1.30,  # left shoulder
            0.00,   # left elbow
            -1.30,  # right shoulder
            0.00,   # right elbow
            -1.10,  # left hip
            0.00,   # left knee
            -0.45,  # left ankle
            -1.10,  # right hip
            0.00,   # right knee
            -0.45,  # right ankle
        ],
        dtype=float,
    )
    maxs = np.array(
        [
            0.50,
            1.30,
            2.10,
            1.30,
            2.10,
            1.10,
            2.10,
            0.75,
            1.10,
            2.10,
            0.75,
        ],
        dtype=float,
    )
    mid = 0.5 * (mins + maxs)
    half = 0.5 * (maxs - mins) * float(scale)
    return mid - half, mid + half


def default_torque_limit(scale: float = 1.0) -> np.ndarray:
    return np.full(N_JOINTS, 120.0 * float(scale), dtype=float)


def oracle_calibration_offsets(descriptor: MorphologyDescriptor) -> np.ndarray:
    """
    Mock ground-truth per-robot calibration offsets.

    In a real project these would come from careful per-robot retargeting tuning.
    Here they are deliberately morphology-dependent so the adapter can learn a
    mock mapping from morphology to correction.
    """
    leg_ratio = descriptor.leg_length / REFERENCE_LEG_LENGTH
    arm_ratio = descriptor.arm_length / REFERENCE_ARM_LENGTH
    mass_ratio = descriptor.total_mass / 65.0

    off = np.zeros(N_JOINTS, dtype=float)

    off[0] = 0.08 * (mass_ratio - 1.0)

    # Arms
    off[1] = 0.18 * (arm_ratio - 1.0)
    off[2] = 0.30 * (arm_ratio - 1.0)
    off[3] = 0.18 * (arm_ratio - 1.0)
    off[4] = 0.30 * (arm_ratio - 1.0)

    # Legs
    off[5] = 0.22 * (leg_ratio - 1.0)
    off[6] = 0.42 * (leg_ratio - 1.0)
    off[7] = 0.14 * (leg_ratio - 1.0)
    off[8] = 0.22 * (leg_ratio - 1.0)
    off[9] = 0.42 * (leg_ratio - 1.0)
    off[10] = 0.14 * (leg_ratio - 1.0)

    return off


def make_robot_model(
    robot_id: str,
    leg_length: float,
    arm_length: float,
    total_mass: float,
    joint_scale: float,
    rng: np.random.Generator,
    noise_std: float,
    oracle_calibration: bool = False,
) -> RobotModel:
    qmin, qmax = default_joint_ranges(joint_scale)
    torque = default_torque_limit(joint_scale)

    descriptor = MorphologyDescriptor(
        robot_id=robot_id,
        leg_length=float(leg_length),
        arm_length=float(arm_length),
        total_mass=float(total_mass),
        joint_range_min=qmin,
        joint_range_max=qmax,
        torque_limit=torque,
        joint_scale=float(joint_scale),
    )

    true_offsets = oracle_calibration_offsets(descriptor)

    if oracle_calibration:
        calibration_offsets = true_offsets.copy()
    else:
        calibration_offsets = true_offsets + rng.normal(0.0, noise_std, size=N_JOINTS)

    # Mock plant distortion: the physical robot behaves as if an unmodeled bias
    # opposes the ideal calibration offsets.
    plant_bias = -true_offsets

    return RobotModel(
        descriptor=descriptor,
        calibration_offsets=calibration_offsets,
        plant_bias=plant_bias,
        license_note="synthetic/mock inline model",
    )


def build_robot_zoo(
    heldout_leg_length: float,
    heldout_arm_length: float,
    heldout_mass: float,
    heldout_joint_scale: float,
    seed: int,
    noise_std: float,
) -> Dict[str, RobotModel]:
    rng = np.random.default_rng(seed + 101)

    robots = {
        "robot_A_train": make_robot_model(
            robot_id="robot_A_train",
            leg_length=0.80,
            arm_length=0.62,
            total_mass=60.0,
            joint_scale=1.00,
            rng=rng,
            noise_std=noise_std,
            oracle_calibration=False,
        ),
        "robot_B_train": make_robot_model(
            robot_id="robot_B_train",
            leg_length=0.92,
            arm_length=0.70,
            total_mass=75.0,
            joint_scale=0.92,
            rng=rng,
            noise_std=noise_std,
            oracle_calibration=False,
        ),
        "robot_C_heldout": make_robot_model(
            robot_id="robot_C_heldout",
            leg_length=heldout_leg_length,
            arm_length=heldout_arm_length,
            total_mass=heldout_mass,
            joint_scale=heldout_joint_scale,
            rng=rng,
            noise_std=0.0,
            oracle_calibration=True,
        ),
    }

    return robots


def make_reference_human_model() -> RobotModel:
    qmin, qmax = default_joint_ranges(1.0)
    torque = default_torque_limit(1.0)
    descriptor = MorphologyDescriptor(
        robot_id="human_reference",
        leg_length=REFERENCE_LEG_LENGTH,
        arm_length=REFERENCE_ARM_LENGTH,
        total_mass=70.0,
        joint_range_min=qmin,
        joint_range_max=qmax,
        torque_limit=torque,
        joint_scale=1.0,
    )
    zeros = np.zeros(N_JOINTS, dtype=float)
    return RobotModel(
        descriptor=descriptor,
        calibration_offsets=zeros,
        plant_bias=zeros,
        license_note="synthetic reference skeleton",
    )


def validate_robot(model: RobotModel) -> List[str]:
    issues: List[str] = []
    d = model.descriptor

    if d.total_mass <= 0:
        issues.append("Total mass must be positive.")
    if d.leg_length <= 0 or d.arm_length <= 0:
        issues.append("Limb lengths must be positive.")
    if np.any(d.joint_range_min >= d.joint_range_max):
        issues.append("Joint range minimum must be below maximum.")
    if np.any(d.torque_limit <= 0):
        issues.append("Torque limits must be positive.")
    if not np.all(np.isfinite(d.vector())):
        issues.append("Morphology descriptor contains non-finite values.")

    # Simplified inertia proxy: positive mass and lengths imply positive scalar
    # inertia proxy in this mock validator.
    inertia_proxy = d.total_mass * (d.leg_length**2 + d.arm_length**2)
    if not np.isfinite(inertia_proxy) or inertia_proxy <= 0:
        issues.append("Inertia proxy is non-positive or non-finite.")

    # Simplified collision check: link fractions must not exceed segment length.
    if LEG_FRACTIONS["thigh"] + LEG_FRACTIONS["shank"] > 1.0 + 1e-6:
        issues.append("Thigh + shank fractions exceed leg length.")
    if ARM_FRACTIONS["upper"] + ARM_FRACTIONS["fore"] > 1.0 + 1e-6:
        issues.append("Upper + fore fractions exceed arm length.")

    return issues


def to_mjcf(model: RobotModel) -> str:
    """
    Generate a simplified mock MJCF string for inspection/download.

    This is not a production-ready robot model. It is included to demonstrate
    robot-export provenance in a self-contained simulation-only app.
    """
    d = model.descriptor
    thigh = d.leg_length * LEG_FRACTIONS["thigh"]
    shank = d.leg_length * LEG_FRACTIONS["shank"]
    upper = d.arm_length * ARM_FRACTIONS["upper"]
    fore = d.arm_length * ARM_FRACTIONS["fore"]
    torso = d.leg_length * LEG_FRACTIONS["torso"]

    return f"""<mujoco model="{d.robot_id}">
  <!-- SIMULATION ONLY / MOCK MJCF / NOT FOR HARDWARE -->
  <option timestep="0.002" integrator="implicitfast"/>
  <compiler angle="radian"/>

  <worldbody>
    <light diffuse=".8 .8 .8" pos="0 0 3" zaxis="0 0 1"/>

    <body name="pelvis" pos="0 0 {d.leg_length:.4f}">
      <freejoint name="root"/>
      <geom type="box" size="0.08 0.06 0.05" rgba="0.2 0.4 0.8 1"/>

      <body name="torso" pos="0 0 {torso:.4f}">
        <joint name="torso_pitch" type="hinge" axis="0 1 0"
               range="{d.joint_range_min[0]:.4f} {d.joint_range_max[0]:.4f}"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.12" radius="0.05" rgba="0.3 0.5 0.9 1"/>

        <body name="left_upper_arm" pos="0 {ARM_FRACTIONS['shoulder_width'] * d.arm_length / 2:.4f} 0.08">
          <joint name="left_shoulder_pitch" type="hinge" axis="0 1 0"
                 range="{d.joint_range_min[1]:.4f} {d.joint_range_max[1]:.4f}"/>
          <geom type="capsule" fromto="0 0 0 0 0 -{upper:.4f}" radius="0.025" rgba="0.4 0.7 0.4 1"/>
          <body name="left_forearm" pos="0 0 -{upper:.4f}">
            <joint name="left_elbow" type="hinge" axis="0 1 0"
                   range="{d.joint_range_min[2]:.4f} {d.joint_range_max[2]:.4f}"/>
            <geom type="capsule" fromto="0 0 0 0 0 -{fore:.4f}" radius="0.022" rgba="0.4 0.7 0.4 1"/>
          </body>
        </body>

        <body name="right_upper_arm" pos="0 -{ARM_FRACTIONS['shoulder_width'] * d.arm_length / 2:.4f} 0.08">
          <joint name="right_shoulder_pitch" type="hinge" axis="0 1 0"
                 range="{d.joint_range_min[3]:.4f} {d.joint_range_max[3]:.4f}"/>
          <geom type="capsule" fromto="0 0 0 0 0 -{upper:.4f}" radius="0.025" rgba="0.9 0.5 0.2 1"/>
          <body name="right_forearm" pos="0 0 -{upper:.4f}">
            <joint name="right_elbow" type="hinge" axis="0 1 0"
                   range="{d.joint_range_min[4]:.4f} {d.joint_range_max[4]:.4f}"/>
            <geom type="capsule" fromto="0 0 0 0 0 -{fore:.4f}" radius="0.022" rgba="0.9 0.5 0.2 1"/>
          </body>
        </body>
      </body>

      <body name="left_thigh" pos="0 {LEG_FRACTIONS['hip_width'] * d.leg_length / 2:.4f} -0.03">
        <joint name="left_hip_pitch" type="hinge" axis="0 1 0"
               range="{d.joint_range_min[5]:.4f} {d.joint_range_max[5]:.4f}"/>
        <geom type="capsule" fromto="0 0 0 0 0 -{thigh:.4f}" radius="0.035" rgba="0.8 0.2 0.2 1"/>
        <body name="left_shank" pos="0 0 -{thigh:.4f}">
          <joint name="left_knee" type="hinge" axis="0 1 0"
                 range="{d.joint_range_min[6]:.4f} {d.joint_range_max[6]:.4f}"/>
          <geom type="capsule" fromto="0 0 0 0 0 -{shank:.4f}" radius="0.030" rgba="0.8 0.2 0.2 1"/>
          <body name="left_foot" pos="0 0 -{shank:.4f}">
            <joint name="left_ankle" type="hinge" axis="0 1 0"
                   range="{d.joint_range_min[7]:.4f} {d.joint_range_max[7]:.4f}"/>
            <geom type="box" size="0.09 0.035 0.02" pos="0.06 0 -0.02" rgba="0.2 0.2 0.2 1"/>
          </body>
        </body>
      </body>

      <body name="right_thigh" pos="0 -{LEG_FRACTIONS['hip_width'] * d.leg_length / 2:.4f} -0.03">
        <joint name="right_hip_pitch" type="hinge" axis="0 1 0"
               range="{d.joint_range_min[8]:.4f} {d.joint_range_max[8]:.4f}"/>
        <geom type="capsule" fromto="0 0 0 0 0 -{thigh:.4f}" radius="0.035" rgba="0.2 0.6 0.8 1"/>
        <body name="right_shank" pos="0 0 -{thigh:.4f}">
          <joint name="right_knee" type="hinge" axis="0 1 0"
                 range="{d.joint_range_min[9]:.4f} {d.joint_range_max[9]:.4f}"/>
          <geom type="capsule" fromto="0 0 0 0 0 -{shank:.4f}" radius="0.030" rgba="0.2 0.6 0.8 1"/>
          <body name="right_foot" pos="0 0 -{shank:.4f}">
            <joint name="right_ankle" type="hinge" axis="0 1 0"
                   range="{d.joint_range_min[10]:.4f} {d.joint_range_max[10]:.4f}"/>
            <geom type="box" size="0.09 0.035 0.02" pos="0.06 0 -0.02" rgba="0.2 0.2 0.2 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


# =============================================================================
# MOTION REPRESENTATION
# =============================================================================


def generate_motion_clip(seed: int, duration: float, fps: int) -> MotionClip:
    """
    Synthetic body-normalized motion generator.

    This is a mock PoseSource / motion generator. It is labelled synthetic in
    the UI and config. No human dataset is accessed.
    """
    rng = np.random.default_rng(seed)
    fps = max(5, int(fps))
    duration = max(0.5, float(duration))
    n = max(2, int(round(duration * fps)))

    t = np.linspace(0.0, duration, n, dtype=float)
    omega = 2.0 * math.pi * 1.15

    root = np.column_stack(
        [
            0.75 * t,
            0.035 * np.sin(omega * t),
            0.95 + 0.035 * np.sin(2.0 * omega * t) + 0.004 * rng.normal(size=n),
        ]
    )

    ankle_norm = np.zeros((n, 2, 3), dtype=float)
    wrist_norm = np.zeros((n, 2, 3), dtype=float)
    foot_contact = np.zeros((n, 2), dtype=bool)

    for side, phase_shift in enumerate([0.0, math.pi]):
        ph = omega * t + phase_shift
        contact = np.cos(ph) > 0.05

        x = 0.26 * np.sin(ph)
        z_stance = -0.95 + 0.015 * np.sin(ph)
        z_swing = -0.76 + 0.09 * np.cos(ph)
        z = np.where(contact, z_stance, z_swing)

        ankle_norm[:, side, 0] = x
        ankle_norm[:, side, 2] = z
        foot_contact[:, side] = contact

    for side, phase_shift in enumerate([math.pi, 0.0]):
        ph = omega * t + phase_shift
        wrist_norm[:, side, 0] = 0.32 * np.sin(ph)
        wrist_norm[:, side, 2] = -0.72 + 0.22 * np.cos(ph)

    confidence = np.clip(0.88 + 0.10 * rng.random(n), 0.0, 1.0)

    return MotionClip(
        t=t,
        root=root,
        wrist_norm=wrist_norm,
        ankle_norm=ankle_norm,
        foot_contact=foot_contact,
        confidence=confidence,
        source="synthetic_mock",
    )


# =============================================================================
# KINEMATICS AND RETARGETING
# =============================================================================


def _ik_two_link(tx: float, tz: float, l1: float, l2: float) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Planar two-link IK in the x-z plane.

    Parent frame:
      +x forward
      +z up
    Default stretched limb points downward: (0, -l1-l2).
    """
    d = math.hypot(tx, tz)
    min_d = abs(l1 - l2) + 1e-4
    max_d = l1 + l2 - 1e-4

    if d < 1e-8:
        dx, dz = 0.0, -1.0
    else:
        dx, dz = tx / d, tz / d

    dc = float(np.clip(d, min_d, max_d))
    tx_c = dx * dc
    tz_c = dz * dc

    cos_q2 = (dc * dc - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    cos_q2 = float(np.clip(cos_q2, -1.0, 1.0))
    q2 = math.acos(cos_q2)

    alpha = math.atan2(tx_c, -tz_c)
    beta = math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
    q1 = alpha - beta

    p1 = np.array([l1 * math.sin(q1), 0.0, -l1 * math.cos(q1)], dtype=float)
    p2 = p1 + np.array([l2 * math.sin(q1 + q2), 0.0, -l2 * math.cos(q1 + q2)], dtype=float)

    return q1, q2, p1, p2


def forward_kinematics(model: RobotModel, q: np.ndarray, root: np.ndarray) -> Dict[str, np.ndarray]:
    d = model.descriptor
    leg = d.leg_length
    arm = d.arm_length

    torso_len = leg * LEG_FRACTIONS["torso"]
    shoulder_width = arm * ARM_FRACTIONS["shoulder_width"]
    hip_width = leg * LEG_FRACTIONS["hip_width"]

    thigh = leg * LEG_FRACTIONS["thigh"]
    shank = leg * LEG_FRACTIONS["shank"]
    foot_len = leg * LEG_FRACTIONS["foot"]

    upper = arm * ARM_FRACTIONS["upper"]
    fore = arm * ARM_FRACTIONS["fore"]

    pelvis = np.asarray(root, dtype=float).copy()
    tp = float(q[0])

    torso_top = pelvis + np.array(
        [torso_len * math.sin(tp), 0.0, torso_len * math.cos(tp)],
        dtype=float,
    )
    head = torso_top + np.array([0.05 * math.sin(tp), 0.0, 0.16], dtype=float)

    pts: Dict[str, np.ndarray] = {
        "pelvis": pelvis,
        "torso_top": torso_top,
        "head": head,
    }

    for side, sign in [("left", 1.0), ("right", -1.0)]:
        pts[f"{side}_shoulder"] = torso_top + np.array([0.0, sign * shoulder_width / 2.0, 0.0])
        pts[f"{side}_hip"] = pelvis + np.array([0.0, sign * hip_width / 2.0, 0.0])

    # Arms
    for side, idx in [("left", 1), ("right", 3)]:
        sp = float(q[idx])
        el = float(q[idx + 1])

        shoulder = pts[f"{side}_shoulder"]
        elbow = shoulder + np.array([upper * math.sin(sp), 0.0, -upper * math.cos(sp)], dtype=float)
        wrist = elbow + np.array([fore * math.sin(sp + el), 0.0, -fore * math.cos(sp + el)], dtype=float)

        pts[f"{side}_elbow"] = elbow
        pts[f"{side}_wrist"] = wrist

    # Legs
    for side, idx in [("left", 5), ("right", 8)]:
        hp = float(q[idx])
        kn = float(q[idx + 1])
        an = float(q[idx + 2])

        hip = pts[f"{side}_hip"]
        knee = hip + np.array([thigh * math.sin(hp), 0.0, -thigh * math.cos(hp)], dtype=float)
        ankle = knee + np.array([shank * math.sin(hp + kn), 0.0, -shank * math.cos(hp + kn)], dtype=float)
        toe = ankle + np.array([foot_len * math.cos(an), 0.0, foot_len * math.sin(an)], dtype=float)

        pts[f"{side}_knee"] = knee
        pts[f"{side}_ankle"] = ankle
        pts[f"{side}_toe"] = toe

    return pts


def simulate_trajectory(
    clip: MotionClip,
    model: RobotModel,
    joint_offsets: np.ndarray,
    use_contact: bool = True,
    plant: bool = True,
    store_points: bool = False,
    direct_mode: bool = False,
) -> TrajectoryResult:
    """
    Mock simulation of retargeting + shared tracking controller.

    The controller is represented here as projected joint targets plus a simple
    plant bias. This is intentionally lightweight and simulation-only.
    """
    t0 = time.perf_counter()
    del t0

    d = model.descriptor
    leg = d.leg_length
    arm = d.arm_length

    thigh = leg * LEG_FRACTIONS["thigh"]
    shank = leg * LEG_FRACTIONS["shank"]
    upper = arm * ARM_FRACTIONS["upper"]
    fore = arm * ARM_FRACTIONS["fore"]

    torso_len = leg * LEG_FRACTIONS["torso"]
    shoulder_width = arm * ARM_FRACTIONS["shoulder_width"]
    hip_width = leg * LEG_FRACTIONS["hip_width"]

    n = clip.num_frames
    dt = float(clip.t[1] - clip.t[0]) if n > 1 else 1.0

    root_scale = leg / REFERENCE_LEG_LENGTH
    roots = clip.root * root_scale

    q_desired = np.zeros((n, N_JOINTS), dtype=float)
    q_actual = np.zeros((n, N_JOINTS), dtype=float)

    ee_error_mm = np.zeros(n, dtype=float)
    foot_skate_mm_s = np.zeros(n, dtype=float)
    fall = np.zeros(n, dtype=bool)
    joint_limit_hit = np.zeros(n, dtype=bool)

    points_frames: Optional[List[Dict[str, np.ndarray]]] = [] if store_points else None

    metric_contact = clip.foot_contact
    prev_ankle: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}

    qmin = d.joint_range_min
    qmax = d.joint_range_max

    for i in range(n):
        ti = float(clip.t[i])
        phase = 2.0 * math.pi * 1.15 * ti

        q_raw = np.zeros(N_JOINTS, dtype=float)

        base_torso = 0.07 * math.sin(phase)
        if direct_mode:
            q_raw[0] = 0.85 * base_torso + 1.25 * float(joint_offsets[0])
        else:
            q_raw[0] = base_torso + float(joint_offsets[0])

        pelvis = roots[i]
        tp = q_raw[0]
        torso_top = pelvis + np.array(
            [torso_len * math.sin(tp), 0.0, torso_len * math.cos(tp)],
            dtype=float,
        )

        targets: Dict[str, np.ndarray] = {}

        # Arms
        for side, idx, sign in [("left", 1, 1.0), ("right", 3, -1.0)]:
            side_index = 0 if side == "left" else 1
            shoulder = torso_top + np.array([0.0, sign * shoulder_width / 2.0, 0.0])
            tn = clip.wrist_norm[i, side_index]

            target = shoulder + np.array(
                [tn[0] * arm, tn[1] * arm, tn[2] * arm],
                dtype=float,
            )
            targets[f"{side}_wrist"] = target

            rel = target - shoulder
            q1, q2, _, _ = _ik_two_link(float(rel[0]), float(rel[2]), upper, fore)

            off1 = float(joint_offsets[idx])
            off2 = float(joint_offsets[idx + 1])

            if direct_mode:
                q1 = 0.85 * q1 + 1.25 * off1
                q2 = 0.85 * q2 + 1.25 * off2
            else:
                q1 += off1
                q2 += off2

            q_raw[idx] = q1
            q_raw[idx + 1] = q2

        # Legs
        for side, idx, sign in [("left", 5, 1.0), ("right", 8, -1.0)]:
            side_index = 0 if side == "left" else 1
            hip = pelvis + np.array([0.0, sign * hip_width / 2.0, 0.0])

            tn = clip.ankle_norm[i, side_index].copy()
            contact = bool(metric_contact[i, side_index])

            if not use_contact:
                # Remove contact-aware stance constraint and inject mock float.
                tn[2] += 0.12 * math.sin(2.0 * phase + (0.0 if side == "left" else math.pi))

            target = hip + np.array([tn[0] * leg, tn[1] * leg, tn[2] * leg], dtype=float)
            targets[f"{side}_ankle"] = target

            rel = target - hip
            q1, q2, _, _ = _ik_two_link(float(rel[0]), float(rel[2]), thigh, shank)

            off1 = float(joint_offsets[idx])
            off2 = float(joint_offsets[idx + 1])
            off3 = float(joint_offsets[idx + 2])

            base_ankle = 0.12 * math.sin(phase + (0.0 if side == "left" else math.pi))

            if direct_mode:
                q1 = 0.85 * q1 + 1.25 * off1
                q2 = 0.85 * q2 + 1.25 * off2
                ankle_q = 0.85 * base_ankle + 1.25 * off3
            else:
                q1 += off1
                q2 += off2
                ankle_q = base_ankle + off3

            q_raw[idx] = q1
            q_raw[idx + 1] = q2
            q_raw[idx + 2] = ankle_q

        q_des_i = np.clip(q_raw, qmin, qmax)
        joint_limit_hit[i] = bool(np.any(np.abs(q_raw - q_des_i) > 1e-5))
        q_desired[i] = q_des_i

        if plant:
            q_phys = q_des_i + model.plant_bias
        else:
            q_phys = q_des_i

        q_act_i = np.clip(q_phys, qmin, qmax)
        q_actual[i] = q_act_i

        pts = forward_kinematics(model, q_act_i, pelvis)

        errors = []
        for key in ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]:
            errors.append(float(np.linalg.norm(pts[key] - targets[key])))
        ee_error_mm[i] = float(np.mean(errors) * 1000.0)

        velocities = []
        for side, side_index in [("left", 0), ("right", 1)]:
            ankle = pts[f"{side}_ankle"]
            if prev_ankle[side] is not None and bool(metric_contact[i, side_index]):
                v = float(np.linalg.norm(ankle - prev_ankle[side]) / max(dt, 1e-6) * 1000.0)
                velocities.append(v)
            prev_ankle[side] = ankle.copy()

        foot_skate_mm_s[i] = float(np.mean(velocities)) if velocities else 0.0

        stance_x: List[float] = []
        stance_y: List[float] = []
        foot_lifted = False

        for side, side_index in [("left", 0), ("right", 1)]:
            if bool(metric_contact[i, side_index]):
                ankle = pts[f"{side}_ankle"]
                toe = pts[f"{side}_toe"]
                stance_x.extend([float(ankle[0]), float(toe[0])])
                stance_y.extend([float(ankle[1]), float(toe[1])])

                # If a nominally contacting foot is too high, treat as instability.
                if float(ankle[2]) > float(pelvis[2]) - 0.86 * leg:
                    foot_lifted = True

        com = pelvis + np.array([0.02 * math.sin(phase), 0.0, 0.0], dtype=float)
        margin = 0.045 * leg

        if not stance_x:
            fall[i] = True
        else:
            x_out = com[0] < min(stance_x) - margin or com[0] > max(stance_x) + margin
            y_out = com[1] < min(stance_y) - margin or com[1] > max(stance_y) + margin
            z_low = com[2] < 0.78 * leg
            fall[i] = bool(x_out or y_out or z_low or foot_lifted)

        if points_frames is not None:
            points_frames.append(pts)

    return TrajectoryResult(
        robot_id=model.descriptor.robot_id,
        q_desired=q_desired,
        q_actual=q_actual,
        root=roots,
        points_frames=points_frames,
        ee_error_mm=ee_error_mm,
        foot_skate_mm_s=foot_skate_mm_s,
        fall=fall,
        joint_limit_hit=joint_limit_hit,
    )


# =============================================================================
# METRICS
# =============================================================================


def make_per_frame_dataframe(
    variant_name: str,
    traj: TrajectoryResult,
    clip: MotionClip,
) -> pd.DataFrame:
    n = traj.ee_error_mm.size
    return pd.DataFrame(
        {
            "variant": variant_name,
            "robot": traj.robot_id,
            "frame": np.arange(n, dtype=int),
            "time_s": clip.t[:n],
            "ee_error_mm": traj.ee_error_mm,
            "foot_skate_mm_s": traj.foot_skate_mm_s,
            "fall": traj.fall,
            "joint_limit_hit": traj.joint_limit_hit,
        }
    )


def summarize_trajectory(variant_name: str, traj: TrajectoryResult) -> Dict[str, Any]:
    ee = float(np.mean(traj.ee_error_mm))
    skate = float(np.mean(traj.foot_skate_mm_s))
    fall_rate = float(np.mean(traj.fall.astype(float)))
    limit_rate = float(np.mean(traj.joint_limit_hit.astype(float)))

    # Demo heuristic thresholds. These are not real safety certification limits.
    success = bool(ee < 150.0 and fall_rate < 0.08 and limit_rate < 0.22)

    return {
        "variant": variant_name,
        "robot": traj.robot_id,
        "ee_error_mm": ee,
        "ee_error_mm_p95": float(np.percentile(traj.ee_error_mm, 95)) if traj.ee_error_mm.size else 0.0,
        "foot_skate_mm_s": skate,
        "fall_rate": fall_rate,
        "joint_limit_hit_rate": limit_rate,
        "success": success,
    }


def safe_metric(df: pd.DataFrame, variant: str, robot: str, col: str, default: float = 0.0) -> float:
    sub = df[(df["variant"] == variant) & (df["robot"] == robot)]
    if sub.empty or col not in sub.columns:
        return float(default)
    return float(sub[col].iloc[0])


# =============================================================================
# BUNDLE / EVALUATION
# =============================================================================


VARIANT_SPECS = [
    ("Baseline IK (uncalibrated)", {"offset_mode": "zero", "use_contact": True, "direct": False}),
    ("Baseline IK (per-robot calibrated)", {"offset_mode": "calibrated", "use_contact": True, "direct": False}),
    ("Morphology adapter (residual)", {"offset_mode": "adapter", "use_contact": True, "direct": False}),
    ("Ablation: no morphology conditioning", {"offset_mode": "mean", "use_contact": True, "direct": False}),
    ("Ablation: no contact flags", {"offset_mode": "adapter", "use_contact": False, "direct": False}),
    ("Ablation: direct output", {"offset_mode": "adapter", "use_contact": True, "direct": True}),
]


def get_offsets(
    mode: str,
    model: RobotModel,
    adapter: LinearMorphologyAdapter,
    mean_offsets: np.ndarray,
) -> np.ndarray:
    if mode == "zero":
        return np.zeros(N_JOINTS, dtype=float)
    if mode == "calibrated":
        return model.calibration_offsets
    if mode == "adapter":
        return adapter.predict(model.descriptor)
    if mode == "mean":
        return mean_offsets
    raise ValueError(f"Unknown offset mode: {mode}")


@st.cache_resource(show_spinner=False)
def build_bundle(
    seed: int,
    duration: float,
    fps: int,
    heldout_leg_length: float,
    heldout_arm_length: float,
    heldout_mass: float,
    heldout_joint_scale: float,
    noise_std: float,
    store_points: bool,
) -> Dict[str, Any]:
    logger.info(
        "Building simulation bundle: seed=%s duration=%s fps=%s store_points=%s",
        seed,
        duration,
        fps,
        store_points,
    )

    robots = build_robot_zoo(
        heldout_leg_length=heldout_leg_length,
        heldout_arm_length=heldout_arm_length,
        heldout_mass=heldout_mass,
        heldout_joint_scale=heldout_joint_scale,
        seed=seed,
        noise_std=noise_std,
    )

    train_ids = ["robot_A_train", "robot_B_train"]
    adapter = LinearMorphologyAdapter.fit(robots, train_ids=train_ids, alpha=1.0)
    mean_offsets = np.mean([robots[rid].calibration_offsets for rid in train_ids], axis=0)

    # Sanity check adapter output shape before using it in simulations.
    probe = adapter.predict(robots["robot_C_heldout"].descriptor)
    if probe.shape != (N_JOINTS,):
        raise RuntimeError(f"Adapter prediction shape wrong: {probe.shape}, expected {(N_JOINTS,)}")

    clip = generate_motion_clip(seed=seed, duration=duration, fps=fps)

    reference_model = make_reference_human_model()
    human_traj = simulate_trajectory(
        clip=clip,
        model=reference_model,
        joint_offsets=np.zeros(N_JOINTS, dtype=float),
        use_contact=True,
        plant=False,
        store_points=store_points,
        direct_mode=False,
    )

    variants: Dict[str, VariantResult] = {}

    for variant_name, spec in VARIANT_SPECS:
        traj_dict: Dict[str, TrajectoryResult] = {}
        per_frame_parts: List[pd.DataFrame] = []
        summary_parts: List[Dict[str, Any]] = []

        for rid, model in robots.items():
            offsets = get_offsets(spec["offset_mode"], model, adapter, mean_offsets)
            traj = simulate_trajectory(
                clip=clip,
                model=model,
                joint_offsets=offsets,
                use_contact=bool(spec["use_contact"]),
                plant=True,
                store_points=store_points,
                direct_mode=bool(spec["direct"]),
            )

            traj_dict[rid] = traj
            per_frame_parts.append(make_per_frame_dataframe(variant_name, traj, clip))
            summary_parts.append(summarize_trajectory(variant_name, traj))

        variants[variant_name] = VariantResult(
            trajectories=traj_dict,
            per_frame=pd.concat(per_frame_parts, ignore_index=True),
            summary=pd.DataFrame(summary_parts),
        )

    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if store_points:
        mins = np.full(3, np.inf, dtype=float)
        maxs = np.full(3, -np.inf, dtype=float)

        def update_bounds(point_dict: Dict[str, np.ndarray]) -> None:
            nonlocal mins, maxs
            for p in point_dict.values():
                p = np.asarray(p, dtype=float)
                mins = np.minimum(mins, p)
                maxs = np.maximum(maxs, p)

        if human_traj.points_frames is not None:
            for pts in human_traj.points_frames:
                update_bounds(pts)

        for vr in variants.values():
            for traj in vr.trajectories.values():
                if traj.points_frames is not None:
                    for pts in traj.points_frames:
                        update_bounds(pts)

        pad = 0.12
        bounds = (mins - pad, maxs + pad)

    config = {
        "seed": int(seed),
        "duration_s": float(duration),
        "fps": int(fps),
        "simulation_only": True,
        "hardware_output_enabled": False,
        "data_source": "synthetic_mock",
        "pose_source": "MockSyntheticPoseSource",
        "robot_zoo": {
            rid: {
                "leg_length_m": float(m.descriptor.leg_length),
                "arm_length_m": float(m.descriptor.arm_length),
                "total_mass_kg": float(m.descriptor.total_mass),
                "joint_scale": float(m.descriptor.joint_scale),
                "license_note": m.license_note,
            }
            for rid, m in robots.items()
        },
        "heldout_morphology": {
            "leg_length_m": float(heldout_leg_length),
            "arm_length_m": float(heldout_arm_length),
            "total_mass_kg": float(heldout_mass),
            "joint_scale": float(heldout_joint_scale),
        },
        "calibration_noise_std": float(noise_std),
        "adapter_fit_seconds": float(adapter.fit_seconds),
        "git_hash": os.environ.get("GIT_HASH", "unknown"),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    validation = {rid: validate_robot(model) for rid, model in robots.items()}

    return {
        "robots": robots,
        "clip": clip,
        "adapter": adapter,
        "mean_offsets": mean_offsets,
        "human_points": human_traj.points_frames,
        "variants": variants,
        "bounds": bounds,
        "config": config,
        "validation": validation,
    }


@st.cache_resource(show_spinner=False)
def build_evaluation(
    heldout_leg_length: float,
    heldout_arm_length: float,
    heldout_mass: float,
    heldout_joint_scale: float,
    noise_std: float,
    seeds: int,
    duration: float,
    fps: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_summary_rows: List[pd.DataFrame] = []

    for s in range(int(seeds)):
        bundle = build_bundle(
            seed=s,
            duration=duration,
            fps=fps,
            heldout_leg_length=heldout_leg_length,
            heldout_arm_length=heldout_arm_length,
            heldout_mass=heldout_mass,
            heldout_joint_scale=heldout_joint_scale,
            noise_std=noise_std,
            store_points=False,
        )

        for variant_name, vr in bundle["variants"].items():
            tmp = vr.summary.copy()
            tmp["seed"] = s
            tmp["variant"] = variant_name
            all_summary_rows.append(tmp)

    raw = pd.concat(all_summary_rows, ignore_index=True)

    metric_cols = ["ee_error_mm", "foot_skate_mm_s", "fall_rate", "joint_limit_hit_rate"]
    agg_rows: List[Dict[str, Any]] = []

    for (variant, robot), grp in raw.groupby(["variant", "robot"]):
        row: Dict[str, Any] = {
            "variant": variant,
            "robot": robot,
            "seeds": int(len(grp)),
        }

        for col in metric_cols:
            vals = grp[col].to_numpy(dtype=float)
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            ci = float(1.96 * std / math.sqrt(vals.size)) if vals.size > 1 else 0.0
            row[col] = mean
            row[f"{col}_ci95"] = ci

        success_vals = grp["success"].astype(float).to_numpy()
        row["success_rate"] = float(np.mean(success_vals))
        row["success_rate_ci95"] = (
            float(1.96 * np.std(success_vals, ddof=1) / math.sqrt(success_vals.size))
            if success_vals.size > 1
            else 0.0
        )

        agg_rows.append(row)

    agg = pd.DataFrame(agg_rows)
    return agg, raw


# =============================================================================
# UNIT CHECKS
# =============================================================================


@st.cache_resource(show_spinner=False)
def run_geometry_checks() -> str:
    # 1) Two-link IK reachability.
    tx, tz = 0.18, -0.72
    l1, l2 = 0.50, 0.40
    _, _, _, p2 = _ik_two_link(tx, tz, l1, l2)
    target = np.array([tx, 0.0, tz], dtype=float)
    err = float(np.linalg.norm(p2 - target))
    if err > 1e-4:
        return f"FAIL: IK reachability error={err:.6e}"

    # 2) Joint-limit projection.
    q = np.array([3.0, -3.0], dtype=float)
    qmin = np.array([-1.0, -1.0], dtype=float)
    qmax = np.array([1.0, 1.0], dtype=float)
    qp = np.clip(q, qmin, qmax)
    if not np.allclose(qp, np.array([1.0, -1.0])):
        return "FAIL: joint-limit projection"

    # 3) Descriptor finiteness.
    mins, maxs = default_joint_ranges(1.0)
    desc = MorphologyDescriptor(
        robot_id="unit_test",
        leg_length=0.8,
        arm_length=0.6,
        total_mass=60.0,
        joint_range_min=mins,
        joint_range_max=maxs,
        torque_limit=default_torque_limit(1.0),
        joint_scale=1.0,
    )
    if not np.all(np.isfinite(desc.vector())):
        return "FAIL: non-finite morphology descriptor"

    # 4) Adapter shape smoke test.
    robots = build_robot_zoo(
        heldout_leg_length=0.74,
        heldout_arm_length=0.55,
        heldout_mass=52.0,
        heldout_joint_scale=1.05,
        seed=0,
        noise_std=0.01,
    )
    adapter = LinearMorphologyAdapter.fit(
        robots,
        train_ids=["robot_A_train", "robot_B_train"],
        alpha=1.0,
    )
    pred = adapter.predict(robots["robot_C_heldout"].descriptor)
    if pred.shape != (N_JOINTS,):
        return f"FAIL: adapter prediction shape {pred.shape}"
    if not np.all(np.isfinite(pred)):
        return "FAIL: adapter prediction non-finite"

    return "PASS"


# =============================================================================
# STREAMLIT VISUALIZATION HELPERS
# =============================================================================


def skeleton_trace(
    points: Dict[str, np.ndarray],
    name: str,
    color: str,
    dash: str = "solid",
    width: int = 5,
) -> go.Scatter3d:
    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    zs: List[Optional[float]] = []

    for a, b in SKELETON_EDGES:
        if a not in points or b not in points:
            continue
        pa = np.asarray(points[a], dtype=float)
        pb = np.asarray(points[b], dtype=float)
        xs.extend([float(pa[0]), float(pb[0]), None])
        ys.extend([float(pa[1]), float(pb[1]), None])
        zs.extend([float(pa[2]), float(pb[2]), None])

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        name=name,
        line=dict(color=color, width=width, dash=dash),
        hoverinfo="skip",
    )


def build_trace_list(
    bundle: Dict[str, Any],
    variant_name: str,
    robot_ids: List[str],
    show_human: bool,
    frame_idx: int,
) -> List[go.Scatter3d]:
    traces: List[go.Scatter3d] = []

    if show_human and bundle["human_points"] is not None:
        traces.append(
            skeleton_trace(
                bundle["human_points"][frame_idx],
                name="Human reference (synthetic)",
                color=ROBOT_COLORS["human_reference"],
                dash="dot",
                width=4,
            )
        )

    variant = bundle["variants"][variant_name]
    for rid in robot_ids:
        traj = variant.trajectories[rid]
        if traj.points_frames is None:
            continue
        traces.append(
            skeleton_trace(
                traj.points_frames[frame_idx],
                name=rid,
                color=ROBOT_COLORS.get(rid, "#333333"),
                dash="solid",
                width=6,
            )
        )

    return traces


def apply_scene_layout(fig: go.Figure, bundle: Dict[str, Any], height: int = 540) -> go.Figure:
    bounds = bundle.get("bounds")
    if bounds is not None:
        mins, maxs = bounds
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[float(mins[0]), float(maxs[0])], title="X forward (m)"),
                yaxis=dict(range=[float(mins[1]), float(maxs[1])], title="Y lateral (m)"),
                zaxis=dict(range=[float(mins[2]), float(maxs[2])], title="Z up (m)"),
                aspectmode="data",
            )
        )
    else:
        fig.update_layout(scene=dict(aspectmode="data"))

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, b=0, t=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_single_frame_figure(
    bundle: Dict[str, Any],
    variant_name: str,
    robot_ids: List[str],
    show_human: bool,
    frame_idx: int,
) -> go.Figure:
    fig = go.Figure(data=build_trace_list(bundle, variant_name, robot_ids, show_human, frame_idx))
    apply_scene_layout(fig, bundle)
    return fig


def build_animated_figure(
    bundle: Dict[str, Any],
    variant_name: str,
    robot_ids: List[str],
    show_human: bool,
) -> go.Figure:
    clip = bundle["clip"]
    n = clip.num_frames

    initial_traces = build_trace_list(bundle, variant_name, robot_ids, show_human, 0)
    frames = []

    for i in range(n):
        frame_traces = build_trace_list(bundle, variant_name, robot_ids, show_human, i)
        frames.append(go.Frame(data=frame_traces, name=str(i)))

    fig = go.Figure(data=initial_traces, frames=frames)

    slider_steps = [
        dict(
            label=f"{i} ({float(clip.t[i]):.2f}s)",
            method="animate",
            args=[
                [str(i)],
                dict(mode="after", frame=dict(duration=40, redraw=True), transition=dict(duration=0)),
            ],
        )
        for i in range(n)
    ]

    sliders = [
        dict(
            active=0,
            pad=dict(b=10, t=40),
            currentvalue=dict(prefix="Frame "),
            steps=slider_steps,
        )
    ]

    updatemenus = [
        dict(
            type="buttons",
            direction="left",
            pad=dict(r=10, t=35),
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=40, redraw=True),
                            transition=dict(duration=0),
                            fromcurrent=True,
                        ),
                    ],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], dict(mode="immediate")],
                ),
            ],
        )
    ]

    apply_scene_layout(fig, bundle, height=620)
    fig.update_layout(sliders=sliders, updatemenus=updatemenus)
    return fig


# =============================================================================
# MAIN APP
# =============================================================================


def main() -> None:
    st.set_page_config(
        page_title="Body-Normalized Humanoid Transfer — Simulation",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    enforce_simulation_only_scope()

    st.title("Body-Normalized Motion Transfer to Humanoid Robots")
    st.caption(
        "**SIMULATION ONLY.** This demo uses synthetic/mock motion and inline mock robot models. "
        "No real robot is commanded, and no real-world result is claimed."
    )

    # -------------------------------------------------------------------------
    # Sidebar configuration
    # -------------------------------------------------------------------------
    st.sidebar.header("Run configuration")

    seed = int(st.sidebar.number_input("Random seed", min_value=0, max_value=10000, value=0, step=1))
    duration = float(st.sidebar.slider("Demo duration (s)", min_value=1.0, max_value=4.0, value=2.0, step=0.1))
    fps = int(st.sidebar.select_slider("Playback FPS", options=[10, 15, 20, 25, 30], value=20))

    st.sidebar.subheader("Held-out morphology sliders")
    heldout_leg_length = float(st.sidebar.slider("Held-out leg length (m)", 0.55, 1.05, 0.74, 0.01))
    heldout_arm_length = float(st.sidebar.slider("Held-out arm length (m)", 0.40, 0.90, 0.55, 0.01))
    heldout_mass = float(st.sidebar.slider("Held-out mass (kg)", 35.0, 100.0, 52.0, 1.0))
    heldout_joint_scale = float(st.sidebar.slider("Held-out joint-range scale", 0.75, 1.25, 1.05, 0.01))
    noise_std = float(st.sidebar.slider("Training calibration noise std", 0.0, 0.05, 0.01, 0.005))

    st.sidebar.subheader("Evaluation")
    eval_seeds = int(st.sidebar.number_input("Evaluation seeds", min_value=5, max_value=10, value=5, step=1))
    run_eval = bool(st.sidebar.checkbox("Auto-run multi-seed evaluation (slower)", value=False))

    st.sidebar.subheader("Hardware output")
    st.sidebar.checkbox(
        "Hardware output (disabled)",
        value=False,
        disabled=True,
        help="Hardware output is off by default and requires an explicit safety-review file.",
    )

    checks = run_geometry_checks()
    st.sidebar.caption(f"Embedded geometry checks: **{checks}**")

    # -------------------------------------------------------------------------
    # Build live playback bundle
    # -------------------------------------------------------------------------
    with st.spinner("Loading synthetic motion, robot zoo, and running simulation-only retargeting..."):
        bundle = build_bundle(
            seed=seed,
            duration=duration,
            fps=fps,
            heldout_leg_length=heldout_leg_length,
            heldout_arm_length=heldout_arm_length,
            heldout_mass=heldout_mass,
            heldout_joint_scale=heldout_joint_scale,
            noise_std=noise_std,
            store_points=True,
        )

    config_hash = hashlib.md5(json.dumps(bundle["config"], sort_keys=True).encode("utf-8")).hexdigest()

    if st.session_state.get("last_config_hash") != config_hash:
        st.session_state.autoplay_done = False
        st.session_state.last_config_hash = config_hash

    if "frame_idx" not in st.session_state:
        st.session_state.frame_idx = 0

    if st.session_state.get("frame_idx", 0) >= bundle["clip"].num_frames:
        st.session_state.frame_idx = 0

    variant_names = list(bundle["variants"].keys())
    preferred_variant = "Morphology adapter (residual)"
    default_variant_index = variant_names.index(preferred_variant) if preferred_variant in variant_names else 0

    robot_ids_all = list(bundle["robots"].keys())
    preferred_robot = "robot_C_heldout"
    default_robot_index = robot_ids_all.index(preferred_robot) if preferred_robot in robot_ids_all else 0
    default_robots = [preferred_robot] if preferred_robot in robot_ids_all else robot_ids_all[:1]

    # -------------------------------------------------------------------------
    # Tabs
    # -------------------------------------------------------------------------
    tab_play, tab_metrics, tab_failures, tab_zoo, tab_logs = st.tabs(
        ["Live playback", "Metrics & ablations", "Failure gallery", "Robot zoo & data", "Logs / export"]
    )

    # -------------------------------------------------------------------------
    # Live playback
    # -------------------------------------------------------------------------
    with tab_play:
        st.subheader("Synchronized human and robot playback")
        st.caption("Auto-play starts once per configuration change. This is a simulated playback, not real hardware.")

        variant_name = st.selectbox(
            "Retargeting method",
            variant_names,
            index=default_variant_index,
            key="playback_variant",
        )

        robot_ids = st.multiselect(
            "Robots to display",
            robot_ids_all,
            default=default_robots,
            key="playback_robots",
        )

        show_human = st.checkbox("Show synthetic human reference", value=True, key="show_human")

        if not robot_ids:
            st.info("Select at least one robot to display.")
        else:
            placeholder = st.empty()
            status = st.empty()

            n_frames = bundle["clip"].num_frames

            if not st.session_state.get("autoplay_done", False):
                st.session_state.autoplay_done = True

                for i in range(n_frames):
                    fig = build_single_frame_figure(bundle, variant_name, robot_ids, show_human, i)
                    placeholder.plotly_chart(fig, use_container_width=True)
                    status.caption(f"Auto-playing simulated frame {i + 1}/{n_frames} — **SIMULATION ONLY**")
                    st.session_state.frame_idx = i
                    time.sleep(1.0 / max(1, fps))

                status.caption("Auto-play complete. Use the interactive animation controls below.")

            frame_idx = st.slider(
                "Frame",
                min_value=0,
                max_value=max(0, n_frames - 1),
                key="frame_idx",
            )

            with st.expander("Interactive Plotly animation", expanded=True):
                anim_fig = build_animated_figure(bundle, variant_name, robot_ids, show_human)
                st.plotly_chart(anim_fig, use_container_width=True)

            if st.button("Replay auto-play"):
                st.session_state.autoplay_done = False
                st.rerun()

    # -------------------------------------------------------------------------
    # Metrics and ablations
    # -------------------------------------------------------------------------
    with tab_metrics:
        st.subheader("Single-seed live simulation metrics")
        single_summary = pd.concat([vr.summary for vr in bundle["variants"].values()], ignore_index=True)
        st.dataframe(single_summary, use_container_width=True)

        st.subheader("Per-frame error traces")

        ts_variant = st.selectbox(
            "Time-series variant",
            variant_names,
            index=default_variant_index,
            key="ts_variant",
        )
        ts_robot = st.selectbox(
            "Time-series robot",
            robot_ids_all,
            index=default_robot_index,
            key="ts_robot",
        )

        df_ts = bundle["variants"][ts_variant].per_frame
        df_ts = df_ts[df_ts["robot"] == ts_robot]

        fig_ts = go.Figure()
        fig_ts.add_trace(
            go.Scatter(
                x=df_ts["time_s"],
                y=df_ts["ee_error_mm"],
                mode="lines",
                name="End-effector error (mm)",
                line=dict(color="#d62728"),
            )
        )
        fig_ts.add_trace(
            go.Scatter(
                x=df_ts["time_s"],
                y=df_ts["foot_skate_mm_s"],
                mode="lines",
                name="Foot skate (mm/s)",
                line=dict(color="#1f77b4", dash="dot"),
                yaxis="y2",
            )
        )
        fig_ts.update_layout(
            height=360,
            xaxis_title="Time (s)",
            yaxis=dict(title="EE error (mm)"),
            yaxis2=dict(title="Foot skate (mm/s)", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        st.subheader("Multi-seed evaluation (simulation-only, deterministic seeds)")

        payload = st.session_state.get("eval_payload")

        if run_eval:
            with st.spinner("Running multi-seed mock evaluation..."):
                eval_agg, eval_raw = build_evaluation(
                    heldout_leg_length=heldout_leg_length,
                    heldout_arm_length=heldout_arm_length,
                    heldout_mass=heldout_mass,
                    heldout_joint_scale=heldout_joint_scale,
                    noise_std=noise_std,
                    seeds=eval_seeds,
                    duration=min(1.6, duration),
                    fps=min(15, fps),
                )
            st.session_state["eval_payload"] = {
                "config_hash": config_hash,
                "agg": eval_agg,
                "raw": eval_raw,
            }
            payload = st.session_state["eval_payload"]
        else:
            if st.button("Run multi-seed evaluation now (slower)"):
                with st.spinner("Running multi-seed mock evaluation..."):
                    eval_agg, eval_raw = build_evaluation(
                        heldout_leg_length=heldout_leg_length,
                        heldout_arm_length=heldout_arm_length,
                        heldout_mass=heldout_mass,
                        heldout_joint_scale=heldout_joint_scale,
                        noise_std=noise_std,
                        seeds=eval_seeds,
                        duration=min(1.6, duration),
                        fps=min(15, fps),
                    )
                st.session_state["eval_payload"] = {
                    "config_hash": config_hash,
                    "agg": eval_agg,
                    "raw": eval_raw,
                }
                payload = st.session_state["eval_payload"]

        if payload is not None and payload.get("config_hash") == config_hash:
            eval_agg = payload["agg"]
            eval_raw = payload["raw"]

            st.dataframe(eval_agg, use_container_width=True)

            held = eval_agg[eval_agg["robot"] == "robot_C_heldout"].copy()
            if not held.empty:
                fig_bar = go.Figure()
                fig_bar.add_trace(
                    go.Bar(
                        x=held["variant"],
                        y=held["ee_error_mm"],
                        name="EE error",
                        error_y=dict(
                            type="data",
                            array=held["ee_error_mm_ci95"],
                            visible=True,
                        ),
                        marker_color="#d62728",
                    )
                )
                fig_bar.update_layout(
                    title="Held-out morphology: end-effector error across seeds",
                    yaxis_title="Mean EE error (mm)",
                    xaxis_title="Method",
                    height=380,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                fig_fall = go.Figure()
                fig_fall.add_trace(
                    go.Bar(
                        x=held["variant"],
                        y=held["fall_rate"],
                        name="Fall rate",
                        error_y=dict(
                            type="data",
                            array=held["fall_rate_ci95"],
                            visible=True,
                        ),
                        marker_color="#ff7f0e",
                    )
                )
                fig_fall.update_layout(
                    title="Held-out morphology: fall rate across seeds",
                    yaxis_title="Fall rate",
                    xaxis_title="Method",
                    height=360,
                )
                st.plotly_chart(fig_fall, use_container_width=True)

            st.download_button(
                label="Download multi-seed raw metrics (CSV)",
                data=eval_raw.to_csv(index=False).encode("utf-8"),
                file_name="multi_seed_raw_metrics.csv",
                mime="text/csv",
            )
        else:
            st.info(
                "Multi-seed evaluation is not enabled by default to keep the live demo fast. "
                "Click the button above or enable auto-run in the sidebar."
            )

        st.subheader("Training / adaptation effort comparison")
        st.caption(
            "The effort numbers below are mock illustrative values for this synthetic demo. "
            "They are not measurements from real hardware or a production training pipeline."
        )

        effort_df = pd.DataFrame(
            [
                {
                    "method": "Classical per-robot IK retargeting",
                    "heldout_ee_error_mm": safe_metric(
                        single_summary,
                        "Baseline IK (uncalibrated)",
                        "robot_C_heldout",
                        "ee_error_mm",
                    ),
                    "per_robot_tuning_effort_h": 2.0,
                    "adaptation_compute_s": 0.0,
                    "notes": "Requires fresh manual tuning for each new morphology.",
                },
                {
                    "method": "Morphology-conditioned adapter",
                    "heldout_ee_error_mm": safe_metric(
                        single_summary,
                        "Morphology adapter (residual)",
                        "robot_C_heldout",
                        "ee_error_mm",
                    ),
                    "per_robot_tuning_effort_h": 0.0,
                    "adaptation_compute_s": float(bundle["adapter"].fit_seconds),
                    "notes": "Mock linear adapter trained on two robots; held-out robot receives predicted residual.",
                },
            ]
        )
        st.dataframe(effort_df, use_container_width=True)

    # -------------------------------------------------------------------------
    # Failure gallery
    # -------------------------------------------------------------------------
    with tab_failures:
        st.subheader("Failure gallery")
        st.caption("Frames are ranked by a mock failure score combining EE error, falls, and joint-limit hits.")

        fail_variant = st.selectbox(
            "Failure-gallery variant",
            variant_names,
            index=default_variant_index,
            key="fail_variant",
        )
        fail_robot = st.selectbox(
            "Failure-gallery robot",
            robot_ids_all,
            index=default_robot_index,
            key="fail_robot",
        )

        df_fail = bundle["variants"][fail_variant].per_frame.copy()
        df_fail = df_fail[df_fail["robot"] == fail_robot]

        if df_fail.empty:
            st.info("No failure data available for this selection.")
        else:
            df_fail["failure_score"] = (
                df_fail["ee_error_mm"]
                + 500.0 * df_fail["fall"].astype(float)
                + 100.0 * df_fail["joint_limit_hit"].astype(float)
            )

            top = df_fail.sort_values("failure_score", ascending=False).head(10)
            st.dataframe(
                top[
                    [
                        "frame",
                        "time_s",
                        "ee_error_mm",
                        "foot_skate_mm_s",
                        "fall",
                        "joint_limit_hit",
                        "failure_score",
                    ]
                ],
                use_container_width=True,
            )

            selected_frame = st.selectbox(
                "Inspect failure frame",
                top["frame"].tolist(),
                key="selected_failure_frame",
            )
            frame_int = int(selected_frame)

            fig_fail = build_single_frame_figure(bundle, fail_variant, [fail_robot], True, frame_int)
            st.plotly_chart(fig_fail, use_container_width=True)

            row = df_fail[df_fail["frame"] == frame_int].iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("EE error", f"{float(row['ee_error_mm']):.1f} mm")
            c2.metric("Foot skate", f"{float(row['foot_skate_mm_s']):.1f} mm/s")
            c3.metric("Fall", str(bool(row["fall"])))
            c4.metric("Joint-limit hit", str(bool(row["joint_limit_hit"])))

    # -------------------------------------------------------------------------
    # Robot zoo and data provenance
    # -------------------------------------------------------------------------
    with tab_zoo:
        st.subheader("Robot zoo")

        zoo_rows = []
        for rid, model in bundle["robots"].items():
            d = model.descriptor
            zoo_rows.append(
                {
                    "robot_id": rid,
                    "role": "held-out" if rid == "robot_C_heldout" else "training",
                    "leg_length_m": d.leg_length,
                    "arm_length_m": d.arm_length,
                    "total_mass_kg": d.total_mass,
                    "joint_scale": d.joint_scale,
                    "mean_torque_Nm": float(np.mean(d.torque_limit)),
                    "validation_issues": "; ".join(bundle["validation"][rid]) or "none",
                    "license_note": model.license_note,
                }
            )

        st.dataframe(pd.DataFrame(zoo_rows), use_container_width=True)

        st.subheader("Mock MJCF exports")
        for rid, model in bundle["robots"].items():
            st.download_button(
                label=f"Download {rid} mock MJCF",
                data=to_mjcf(model),
                file_name=f"{rid}.mock.mjcf",
                mime="application/xml",
            )

        st.subheader("Motion data provenance")
        st.json(
            {
                "source": bundle["clip"].source,
                "license": "synthetic/mock generated in-app",
                "real_human_dataset_used": False,
                "mediapipe_used": False,
                "wham_or_4dhumans_used": False,
                "split_policy": "not applicable for synthetic mock clip",
                "confidence_mean": float(np.mean(bundle["clip"].confidence)),
                "notes": "A production system would use consented/licensed datasets such as AMASS/LaFAN1 subsets or a labelled synthetic generator.",
            }
        )

        st.subheader("Interfaces represented in this demo")
        st.markdown(
            """
            - `PoseSource`: mock synthetic motion generator.
            - `MotionRepresentation`: root trajectory, normalized wrist/ankle targets, foot-contact flags, confidence.
            - `Retargeter`: IK baseline plus morphology-conditioned residual adapter.
            - `Controller`: simplified projected joint-target tracking with joint-limit projection.
            - `SafetyLayer`: simulation-only guard; hardware output disabled by default.
            - `Metrics`: EE error, foot skate, fall rate, joint-limit hits, success heuristic.
            - `Logger`: config and CSV exports.
            """
        )

    # -------------------------------------------------------------------------
    # Logs and export
    # -------------------------------------------------------------------------
    with tab_logs:
        st.subheader("Run manifest")
        st.json(bundle["config"])

        st.download_button(
            label="Download run config (JSON)",
            data=json.dumps(bundle["config"], indent=2),
            file_name="run_config.json",
            mime="application/json",
        )

        st.subheader("Metrics export")

        all_summary = pd.concat([vr.summary for vr in bundle["variants"].values()], ignore_index=True)
        st.download_button(
            label="Download summary metrics (CSV)",
            data=all_summary.to_csv(index=False).encode("utf-8"),
            file_name="summary_metrics.csv",
            mime="text/csv",
        )

        export_variant = st.selectbox(
            "Variant for per-frame CSV export",
            variant_names,
            index=default_variant_index,
            key="export_variant",
        )
        export_robot = st.selectbox(
            "Robot for per-frame CSV export",
            robot_ids_all,
            index=default_robot_index,
            key="export_robot",
        )

        export_df = bundle["variants"][export_variant].per_frame
        export_df = export_df[export_df["robot"] == export_robot]

        st.download_button(
            label="Download selected per-frame metrics (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"per_frame_{export_robot}_{hashlib.md5(export_variant.encode()).hexdigest()[:8]}.csv",
            mime="text/csv",
        )

        st.subheader("Safety statement")
        st.markdown(
            """
            This application is **simulation only**. It does not publish commands to real hardware.
            Hardware output is disabled by default and would require an explicit safety-review file.
            All displayed motions, robots, and metrics in this demo are synthetic/mock and should not be
            interpreted as real-robot performance results.
            """
        )


if __name__ == "__main__":
    main()