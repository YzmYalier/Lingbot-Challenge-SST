#!/usr/bin/env python3
"""Render pre-flight for the RoboTwin sim side.

A marginally-provisioned Vulkan stack survives a toy 64x64 single-camera scene
and then wedges under the real load, so this mirrors what an episode actually
drives: three cameras at the eval resolution over many frames, with real
geometry for the rasteriser to chew on.

Without this check a broken stack is not obvious -- the eval loop swallows the
render exception and spins through seeds forever, so the run looks alive while
producing nothing.

Progress is flushed every iteration: when the caller's `timeout` kills us, the
last line still says how far we got. A hang and a crash would otherwise look
identical (no output at all).
"""
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import sapien  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
W = int(sys.argv[2]) if len(sys.argv) > 2 else 384
H = int(sys.argv[3]) if len(sys.argv) > 3 else 320

print(f"[render_check] {N} frames, 3 cameras @ {W}x{H}", flush=True)
t0 = time.time()

try:
    # Build the scene exactly the way RoboTwin's _base_task.setup_scene does:
    # Engine + SapienRenderer, the same global material/texture limits, and above
    # all the ray-tracing camera shader it selects. This is not cosmetic. With
    # SAPIEN's *default* (rasterization) shader dir, get_picture("Color") blocks
    # forever on this node, so a bare sapien.Scene() check hangs and reports the
    # stack as dead while real episodes render fine at ~10 fps -- a false
    # negative that aborts the whole run at pre-flight.
    engine = sapien.Engine()
    from sapien.render import set_global_config

    set_global_config(max_num_materials=50000, max_num_textures=50000)
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    sapien.render.set_camera_shader_dir("rt")
    sapien.render.set_ray_tracing_samples_per_pixel(32)
    sapien.render.set_ray_tracing_path_depth(8)
    sapien.render.set_ray_tracing_denoiser("oidn")

    scene = engine.create_scene(sapien.SceneConfig())
    scene.set_timestep(1 / 250)
    scene.add_ground(0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 0.5, -1], [0.5, 0.5, 0.5])
    for i in range(24):
        b = scene.create_actor_builder()
        b.add_box_collision(half_size=[0.05, 0.05, 0.05])
        b.add_box_visual(half_size=[0.05, 0.05, 0.05], material=[0.6, 0.3, 0.3])
        a = b.build(name=f"box{i}")
        a.set_pose(sapien.Pose([0.3 * (i % 6) - 0.75, 0.3 * (i // 6) - 0.45, 0.06]))
    cams = [scene.add_camera(f"cam{i}", W, H, 1.0, 0.1, 100) for i in range(3)]
    for i, c in enumerate(cams):
        c.set_pose(sapien.Pose([-1.0 - 0.2 * i, 0.3 * i, 0.8]))
except Exception as exc:  # noqa: BLE001
    print(f"[render_check] FAILED during scene setup: {exc}", flush=True)
    sys.exit(1)

for i in range(N):
    try:
        scene.step()
        scene.update_render()
        for c in cams:
            c.take_picture()
            c.get_picture("Color")
    except Exception as exc:  # noqa: BLE001
        print(f"[render_check] FAILED at frame {i + 1}/{N}: {exc}", flush=True)
        sys.exit(1)
    if (i + 1) % 20 == 0 or i == 0:
        print(f"[render_check] frame {i + 1}/{N}  {time.time() - t0:.1f}s", flush=True)

dt = time.time() - t0
print(f"[render_check] OK: {N}/{N} frames x3 cams in {dt:.1f}s ({N / dt:.1f} fps)", flush=True)
