# Calibrated cameras of the real G1-D

`g1d_cameras_20261027.json` and `camera_lens.py` are Adam Basl's calibration of the lab's Unitree G1-D head camera
(left view of the stereo pair) and both Dex1-1 wrist cameras (2026-10-27), copied unchanged. The JSON holds each
lens (OpenCV K, D), its pose in the parent link of the G1-D URDF and the Isaac render settings; `LensWarp` turns the
centred pinhole render into the real lens (focal lengths, principal point, distortion).

In this task (`env_cfg.py: calibrated_camera_cfgs`, `G1-Dex1-Two-Tables-Camera-v0`):
- The head camera is mounted on `torso_link` with the calibrated pose.
- The wrist cameras are mounted on `{side}_wrist_yaw_link`, at the calibrated pose plus 0.0415 m along x. In the G1-D URDF,
  `{side}_dex1_base_link` is the wrist yaw link moved by that amount, not turned; this model's `{side}_hand_base_link`
  is turned 90 deg. Torso and arm kinematics of this model and the G1-D URDF agree.
- `env.camera_images()` returns the three 640 x 480 images as the real cameras see them.

Valid for the lab's G1-D as mounted on 2026-10-27: remounting a wrist camera or replacing the head unit invalidates
the poses.
