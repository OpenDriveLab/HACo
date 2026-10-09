# SharpA hand kinematics

These URDFs are derived from
[sharpa-robotics/sharpa-urdf-usd-xml](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/tree/0d19cac602f46456b819e4b6a2c09a74982c9a3e),
commit `0d19cac602f46456b819e4b6a2c09a74982c9a3e`.
Copyright 2025 Sharpa Group; licensed under Apache 2.0.
The upstream license and notice are included in this directory.

HACo removes visual and collision meshes and MuJoCo directives because its
skeleton renderer uses forward kinematics only. All links, joint transforms,
axes, limits, and inertial properties are preserved. These files provide the
same hand keypoints as the original models without external mesh files.

The files ship with the repository and Python package. The runtime uses them
by default; `SHARPA_URDF_DIR` can override the asset directory when needed.
