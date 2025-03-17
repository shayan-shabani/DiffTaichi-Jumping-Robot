import sys
import taichi as ti
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['figure.dpi'] = 150

real = ti.f32
ti.init(default_fp=real)

max_steps = 8192
vis_interval = 256
output_vis_interval = 16
steps = 4096
assert steps * 2 <= max_steps

vis_resolution = 1024

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(2, dtype=real)

loss = scalar()

use_toi = False

x = vec()
v = vec()
rotation = scalar()
omega = scalar()

halfsize = vec()
inverse_mass = scalar()
inverse_inertia = scalar()

v_inc = vec()
x_inc = vec()
rotation_inc = scalar()
omega_inc = scalar()

head_id = 1
goal = vec()


ground_height = 0.1
top_height = 1.0
gravity = -11
friction = 0.3
penalty = 1e4
damping = 3.0
elasticity = 0.0
lambda_reg = 200
land_same_position_weight = 100.0

jump_completed = 0  # 0: not landed yet, 1: landed and reset completed
reset_counter = 0
reset_duration = 100  # number of time steps over which to interpolate
landing_threshold = ground_height + 0.05  # when head is near ground


gradient_clip = 30
spring_omega = 30
default_actuation = 0.05

n_springs = 0
spring_anchor_a = ti.field(ti.i32)
spring_anchor_b = ti.field(ti.i32)
spring_length = scalar()
spring_offset_a = vec()
spring_offset_b = vec()
spring_phase = scalar()
spring_actuation = scalar()
spring_stiffness = scalar()
nominal_spring_length = scalar()


n_sin_waves = 10
n_hidden = 32
weights1 = scalar()
bias1 = scalar()
hidden = scalar()
weights2 = scalar()
bias2 = scalar()
actuation = scalar()


def n_input_states():
    return n_sin_waves + 6 * n_objects + 2


def allocate_fields():
    ti.root.dense(ti.i,
                  max_steps).dense(ti.j,
                                   n_objects).place(x, v, rotation,
                                                    rotation_inc, omega, v_inc,
                                                    x_inc, omega_inc)
    ti.root.dense(ti.i, n_objects).place(halfsize, inverse_mass,
                                         inverse_inertia)
    ti.root.dense(ti.i, n_springs).place(spring_anchor_a, spring_anchor_b,
                                         spring_length, spring_offset_a,
                                         spring_offset_b, spring_stiffness,
                                         spring_phase, spring_actuation)
    ti.root.dense(ti.ij, (n_hidden, n_input_states())).place(weights1)
    ti.root.dense(ti.ij, (n_springs, n_hidden)).place(weights2)
    ti.root.dense(ti.i, n_hidden).place(bias1)
    ti.root.dense(ti.i, n_springs).place(bias2)
    ti.root.dense(ti.ij, (max_steps, n_springs)).place(actuation)
    ti.root.dense(ti.ij, (max_steps, n_hidden)).place(hidden)
    ti.root.dense(ti.i, n_springs).place(nominal_spring_length)
    ti.root.place(loss, goal)
    ti.root.lazy_grad()


dt = 0.00001
learning_rate = 0.01


@ti.func
def rotation_matrix(r):
    return ti.Matrix([[ti.cos(r), -ti.sin(r)], [ti.sin(r), ti.cos(r)]])


@ti.kernel
def initialize_properties():
    for i in range(n_objects):
        inverse_mass[i] = 1.0 / (4 * halfsize[i][0] * halfsize[i][1])
        inverse_inertia[i] = 1.0 / (
                4 / 3 * halfsize[i][0] * halfsize[i][1] *
                                    (halfsize[i][0] * halfsize[i][0] +
                                     halfsize[i][1] * halfsize[i][1]))


@ti.func
def to_world(t, i, rela_x):
    rot = rotation[t, i]
    rot_matrix = rotation_matrix(rot)
    rela_pos = rot_matrix @ rela_x
    rela_v = omega[t, i] * ti.Vector([-rela_pos[1], rela_pos[0]])

    world_x = x[t, i] + rela_pos
    world_v = v[t, i] + rela_v

    return world_x, world_v, rela_pos


@ti.func
def apply_impulse(t, i, impulse, location, toi_input):
    delta_v = impulse * inverse_mass[i]
    delta_omega = (location - x[t, i]).cross(impulse) * inverse_inertia[i]

    toi = ti.min(ti.max(0.0, toi_input), dt)
    ti.atomic_add(x_inc[t + 1, i], toi * (-delta_v))
    ti.atomic_add(rotation_inc[t + 1, i], toi * (-delta_omega))

    ti.atomic_add(v_inc[t + 1, i], delta_v)
    ti.atomic_add(omega_inc[t + 1, i], delta_omega)


@ti.kernel
def collide(t: ti.i32):
    for i in range(n_objects):
        hs = halfsize[i]
        for k in ti.static(range(4)):
            # the corner for collision detection
            offset_scale = ti.Vector([k % 2 * 2 - 1, k // 2 % 2 * 2 - 1])

            corner_x, corner_v, rela_pos = to_world(t, i, offset_scale * hs)
            corner_v = corner_v + dt * gravity * ti.Vector([0.0, 1.0])

            # Apply impulse so that there's no sinking
            normal = ti.Vector([0.0, 1.0])
            tao = ti.Vector([1.0, 0.0])

            rn = rela_pos.cross(normal)
            rt = rela_pos.cross(tao)
            impulse_contribution = inverse_mass[i] + (rn) ** 2 * \
                                   inverse_inertia[i]
            timpulse_contribution = inverse_mass[i] + (rt) ** 2 * \
                                    inverse_inertia[i]

            rela_v_ground = normal.dot(corner_v)

            impulse = 0.0
            timpulse = 0.0
            new_corner_x = corner_x + dt * corner_v
            toi = 0.0
            if rela_v_ground < 0 and new_corner_x[1] < ground_height:
                impulse = -(1 +
                            elasticity) * rela_v_ground / impulse_contribution
                if impulse > 0:
                    # friction
                    timpulse = -corner_v.dot(tao) / timpulse_contribution
                    timpulse = ti.min(friction * impulse,
                                      ti.max(-friction * impulse, timpulse))
                    if corner_x[1] > ground_height:
                        toi = -(corner_x[1] - ground_height) / ti.min(
                            corner_v[1], -1e-3)

                apply_impulse(t, i, impulse * normal + timpulse * tao,
                          new_corner_x, toi)

            if corner_v[1] > 0 and new_corner_x[1] > top_height:
                # normal for top boundary is downward
                normal = ti.Vector([0.0, -1.0])
                tao = ti.Vector([1.0, 0.0])

                rn = rela_pos.cross(normal)
                rt = rela_pos.cross(tao)
                impulse_contribution = inverse_mass[i] + (rn) ** 2 * inverse_inertia[i]
                timpulse_contribution = inverse_mass[i] + (rt) ** 2 * inverse_inertia[i]

                # velocity relative to the 'ceiling' normal
                rela_v_top = normal.dot(corner_v)

                impulse = -(1 + elasticity) * rela_v_top / impulse_contribution
                if impulse > 0:
                    # friction
                    timpulse = -corner_v.dot(tao) / timpulse_contribution
                    timpulse = ti.min(friction * impulse, ti.max(-friction * impulse, timpulse))

                apply_impulse(t, i, impulse * normal + timpulse * tao, new_corner_x, 0.0)

            penalty_force = 0.0
            if new_corner_x[1] < ground_height:
                # apply penalty
                penalty_force = -dt * penalty * (
                    new_corner_x[1] - ground_height) / impulse_contribution

            apply_impulse(t, i, penalty_force * normal, new_corner_x, 0)


@ti.kernel
def apply_spring_force(t: ti.i32):
    for i in range(n_springs):
        a = spring_anchor_a[i]
        b = spring_anchor_b[i]
        pos_a, vel_a, rela_a = to_world(t, a, spring_offset_a[i])
        pos_b, vel_b, rela_b = to_world(t, b, spring_offset_b[i])
        dist = pos_a - pos_b
        length = dist.norm() + 1e-4

        is_joint = (spring_length[i] == -1)
        # Identify leg springs by their nominal rest length (e.g. knee springs should be longer than 0.05)
        is_leg_spring = False
        if not is_joint and (nominal_spring_length[i] > 0.05):
            is_leg_spring = True

        # Compute effective target length as before:
        effective_target_length = 0.0
        if (not is_joint) and (spring_actuation[i] != 0):
            act_signal = ti.sin(2 * math.pi * (t * dt))
            effective_target_length = spring_length[i] - spring_actuation[i] * act_signal * 0.01
        else:
            if is_joint:
                effective_target_length = 0.0
            else:
                effective_target_length = spring_length[i]

        # Compute the magnitude of the spring force:
        impulse_magnitude = dt * (length - effective_target_length) * spring_stiffness[i] / length

        # Declare impulse_dir first
        impulse_dir = ti.Vector([0.0, 0.0])
        # For leg springs, lock the force direction to vertical:
        if is_leg_spring:
            impulse_dir = ti.Vector([0.0, 1.0])
        else:
            impulse_dir = dist / (dist.norm() + 1e-4)

        impulse = impulse_magnitude * impulse_dir

        if is_joint:
            rela_vel = vel_a - vel_b
            rela_vel_norm = rela_vel.norm() + 1e-1
            impulse_dir_2 = rela_vel / rela_vel_norm
            impulse_contribution = (inverse_mass[a] +
                                    impulse_dir_2.cross(rela_a) ** 2 * inverse_inertia[a] +
                                    inverse_mass[b] +
                                    impulse_dir_2.cross(rela_b) ** 2 * inverse_inertia[b])
            impulse += rela_vel_norm / impulse_contribution * impulse_dir_2

        # Add damping force
        damping_coef = 0.2
        dv = vel_a - vel_b
        max_dv = 50.0
        if dv.norm() > max_dv:
            dv = dv.normalized() * max_dv
        damping_force = -damping_coef * dv
        impulse += dt * damping_force

        max_impulse = 1e4
        if impulse.norm() > max_impulse:
            impulse *= max_impulse / impulse.norm()

        apply_impulse(t, a, -impulse, pos_a, 0.0)
        apply_impulse(t, b, impulse, pos_b, 0.0)

@ti.kernel
def advance_no_toi(t: ti.i32):
    for i in range(n_objects):
        s = math.exp(-dt * damping)
        v[t, i] = s * v[t - 1, i] + v_inc[t, i] + dt * gravity * ti.Vector([0.0, 1.0])
        # For head (object head_id), clamp if next position would exceed top_height.
        if i == head_id:
            predicted_y = x[t - 1, i].y + dt * v[t, i].y
            if predicted_y > top_height:
                v[t, i] = ti.Vector([v[t, i][0], 0.0])
        x[t, i] = x[t - 1, i] + dt * v[t, i]
        if i == 0:
            omega[t, i] = 0.0
            rotation[t, i] = rotation[0, i]
        else:
            if x[t, i].y > ground_height + 0.05:
                omega[t, i] = 0.0
            else:
                omega[t, i] = s * omega[t - 1, i] + omega_inc[t, i]
            rotation[t, i] = rotation[t - 1, i] + dt * omega[t, i]

@ti.kernel
def reset_loss():
    loss[None] = 0.0

@ti.kernel
def compute_head_height_loss():
    final_y = x[steps - 1, head_id].y
    initial_y = x[0, head_id].y
    loss[None] += -(final_y - initial_y)

@ti.kernel
def compute_torso_height_penalty():
    loss[None] += 100.0 * ti.max(0.0, x[steps - 1, 0].y - (ground_height + 0.2))**2

@ti.kernel
def compute_velocity_penalty():
    for i in range(n_objects):
        loss[None] += 10.0 * v[steps - 1, i].norm_sqr()

@ti.kernel
def compute_geometry_regularization():
    for i in range(n_springs):
        a = spring_anchor_a[i]
        b = spring_anchor_b[i]
        pos_a = x[steps - 1, a]
        pos_b = x[steps - 1, b]
        current_length = (pos_a - pos_b).norm() + 1e-4
        loss[None] += lambda_reg * (current_length - nominal_spring_length[i])**2

@ti.kernel
def compute_landing_penalty():
    for i in range(n_objects):
        diff = x[steps - 1, i] - x[0, i]
        loss[None] += land_same_position_weight * diff.norm_sqr()


@ti.kernel
def compute_velocity_penalty_final():
    for i in range(n_objects):
        loss[None] += 2.0 * v[steps - 1, i].norm_sqr()

@ti.kernel
def compute_torso_rotation_penalty():
    loss[None] += 50.0 * (rotation[steps - 1, 0]**2)


@ti.kernel
def compute_leg_spring_rotation_penalty():
    # Penalize the difference in rotation between the thigh and calf for both legs.
    loss[None] += 50.0 * (rotation[steps - 1, 2] - rotation[steps - 1, 3])**2
    loss[None] += 50.0 * (rotation[steps - 1, 4] - rotation[steps - 1, 5])**2


@ti.kernel
def clamp_hip_rotation():
    t = steps - 1
    for i in ti.static([2, 4]):
        rel = rotation[t, 0] - rotation[t, i]  # torso (object 0) minus thigh
        max_rel = 0.35  # maximum allowed deviation in radians (~20°)
        if rel > max_rel:
            rotation[t, i] = rotation[t, 0] - max_rel
            omega[t, i] = 0.0  # reset angular velocity for stability
        elif rel < -max_rel:
            rotation[t, i] = rotation[t, 0] + max_rel
            omega[t, i] = 0.0

def compute_full_loss():
    reset_loss()
    compute_head_height_loss()
    compute_torso_height_penalty()
    compute_velocity_penalty()
    compute_torso_rotation_penalty()
    compute_geometry_regularization()
    compute_landing_penalty()
    compute_velocity_penalty_final()

gui = ti.GUI('Rigid Body Simulation', (512, 512), background_color=0xFFFFFF)

apex_threshold = 0.9  # When head.y > 0.9 and head is falling, we reset

# Global flag to indicate whether we have reset for the current jump.
has_reset = ti.field(dtype=ti.i32, shape=())  # 0: not reset, 1: reset done

@ti.kernel
def reset_lower_body(t: ti.i32):
    # Reset lower body objects (legs) to their initial positions and zero their velocities.
    for i in ti.static([2, 3, 4, 5]):
        x[t, i] = x[0, i]
        v[t, i] = ti.Vector([0.0, 0.0])
        rotation[t, i] = rotation[0, i]
        omega[t, i] = 0.0

@ti.kernel
def reset_spring_lengths():
    # Reset spring rest lengths for non-joint springs.
    for i in range(n_springs):
        if spring_length[i] != -1:
            spring_length[i] = nominal_spring_length[i]

@ti.kernel
def smooth_reset_lower_body(t: ti.i32, alpha: real):
    for i in ti.static([2, 3, 4, 5]):
        x[t, i] = (1 - alpha) * x[t, i] + alpha * x[0, i]
        v[t, i] = (1 - alpha) * v[t, i]
        rotation[t, i] = (1 - alpha) * rotation[t, i] + alpha * rotation[0, i]
        omega[t, i] = (1 - alpha) * omega[t, i]


def forward(output=None, visualize=True):
    initialize_properties()
    global jump_completed
    jump_completed = 0

    interval = vis_interval
    total_steps = steps
    if output:
        print(output)
        interval = output_vis_interval
        os.makedirs('rigid_body/{}/'.format(output), exist_ok=True)
        total_steps *= 2

    goal[None] = [0.9, 0.15]
    for t in range(1, total_steps):
        if jump_completed == 0:
            collide(t - 1)
            apply_spring_force(t - 1)
            advance_no_toi(t)

            if x[t, head_id].y < landing_threshold and v[t, head_id].y <= 0:
                # Freeze the state: zero velocities for all objects.
                for i in range(n_objects):
                    v[t, i] = ti.Vector([0.0, 0.0])
                    omega[t, i] = 0.0
                jump_completed = 1  # mark that we've landed and freeze further updates.
        else:
            for i in range(n_objects):
                x[t, i] = x[t - 1, i]
                v[t, i] = ti.Vector([0.0, 0.0])
                rotation[t, i] = rotation[t - 1, i]
                omega[t, i] = 0.0

        if (t + 1) % interval == 0 and visualize:
            for i in range(n_objects):
                points = []
                for k in range(4):
                    offset_scale = [[-1, -1], [1, -1], [1, 1], [-1, 1]][k]
                    rot = rotation[t, i]
                    rot_matrix = np.array([[math.cos(rot), -math.sin(rot)],
                                           [math.sin(rot), math.cos(rot)]])
                    pos = np.array([x[t, i][0], x[t, i][1]]) + offset_scale * (rot_matrix @ np.array([halfsize[i][0], halfsize[i][1]]))
                    points.append((pos[0], pos[1]))
                for k in range(4):
                    gui.line(points[k], points[(k + 1) % 4], color=0x0, radius=2)
            for i in range(n_springs):
                def get_world_loc(i, offset):
                    rot = rotation[t, i]
                    rot_matrix = np.array([[math.cos(rot), -math.sin(rot)],
                                           [math.sin(rot), math.cos(rot)]])
                    pos = np.array([[x[t, i][0]], [x[t, i][1]]]) + rot_matrix @ np.array([[offset[0]], [offset[1]]])
                    return pos
                pt1 = get_world_loc(spring_anchor_a[i], spring_offset_a[i])
                pt2 = get_world_loc(spring_anchor_b[i], spring_offset_b[i])
                color = 0xFF2233
                if spring_actuation[i] != 0 and spring_length[i] != -1:
                    a = actuation[t - 1, i] * 0.5
                    color = ti.rgb_to_hex((0.5 + a, 0.5 - abs(a), 0.5 - a))
                if spring_length[i] == -1:
                    gui.line(pt1, pt2, color=0x000000, radius=9)
                    gui.line(pt1, pt2, color=color, radius=7)
                else:
                    gui.line(pt1, pt2, color=0x000000, radius=7)
                    gui.line(pt1, pt2, color=color, radius=5)
            gui.line((0.05, ground_height - 5e-3), (0.95, ground_height - 5e-3), color=0x0, radius=5)
            file = None
            if output:
                file = f'rigid_body/{output}/{t:04d}.png'
            gui.show(file=file)
    loss[None] = 0
    clamp_hip_rotation()
    # Include the new penalty to minimize relative rotation in leg springs:
    compute_leg_spring_rotation_penalty()
    compute_full_loss()


@ti.kernel
def clear_states():
    for t in range(0, max_steps):
        for i in range(0, n_objects):
            v_inc[t, i] = ti.Vector([0.0, 0.0])
            x_inc[t, i] = ti.Vector([0.0, 0.0])
            rotation_inc[t, i] = 0.0
            omega_inc[t, i] = 0.0


def setup_robot(objects, springs, h_id):
    global head_id
    head_id = h_id
    global n_objects, n_springs
    n_objects = len(objects)
    n_springs = len(springs)
    allocate_fields()

    print('n_objects=', n_objects, '   n_springs=', n_springs)

    for i in range(n_objects):
        x[0, i] = objects[i][0]
        halfsize[i] = objects[i][1]
        rotation[0, i] = objects[i][2]

    for i in range(n_springs):
        s = springs[i]
        spring_anchor_a[i] = s[0]
        spring_anchor_b[i] = s[1]
        spring_offset_a[i] = s[2]
        spring_offset_b[i] = s[3]
        spring_length[i]   = s[4]
        nominal_spring_length[i] = s[4]
        spring_stiffness[i]= s[5]
        if s[6]:
            spring_actuation[i] = s[6]
        else:
            spring_actuation[i] = default_actuation


def optimize_jump(toi=True, visualize=True):
    global use_toi
    use_toi = toi

    losses = []
    stiffness_values = []

    # Knee springs are now at indices 3 and 4
    knee_springs = [3, 4]


    # Initialize parameters for the knee springs
    for sid in knee_springs:
        spring_length[sid] = 0.2
        spring_stiffness[sid] = 200.0  # initial stiffness

    n_iterations = 20
    lr_stiffness = 1e-7  # learning rate for stiffness adjustments
    max_grad = 1e3  # gradient clipping threshold

    params = [(sid, 'stiffness') for sid in knee_springs]

    for it in range(n_iterations):
        clear_states()
        with ti.ad.Tape(loss):
            forward(visualize=False)

        current_loss = float(loss[None])
        losses.append(current_loss)
        stiffness_values.append([float(spring_stiffness[sid]) for sid in knee_springs])

        print(f'Iter={it}, Loss={current_loss}')

        for sid, param_type in params:
            g = spring_stiffness.grad[sid]
            clipped_g = max(-max_grad, min(max_grad, g))
            spring_stiffness[sid] -= lr_stiffness * clipped_g
            spring_stiffness[sid] = max(100.0, min(1000.0, spring_stiffness[sid]))

        print(f'Knee spring (idx {knee_springs[0]}): L={spring_length[knee_springs[0]]:.4f}, K={spring_stiffness[knee_springs[0]]:.1f} | '
              f'Knee spring (idx {knee_springs[1]}): L={spring_length[knee_springs[1]]:.4f}, K={spring_stiffness[knee_springs[1]]:.1f}')

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('Training Progress')
    plt.ylabel('Loss')
    plt.grid(True)

    # Plot stiffness values
    plt.subplot(2, 1, 2)
    plt.plot(stiffness_values, linewidth=2)
    plt.title('Spring Stiffness Evolution')
    plt.xlabel('Iterations')
    plt.ylabel('Stiffness')
    plt.legend([f'Spring {sid}' for sid in knee_springs])
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.savefig('training_progress.png')
    print("Saved training plot to training_progress.png")

    clear_states()
    forward("final_jump", visualize=True)




robot_id = 0
if len(sys.argv) != 3:
    print(
        "Usage: python3 rigid_body.py [robot_id=0, 1, 2, ...] [cmd=train/plot]"
    )
    exit(-1)
else:
    robot_id = int(sys.argv[1])
    cmd = sys.argv[2]
print(robot_id, cmd)


def main():
    robot_id = int(sys.argv[1])
    cmd = sys.argv[2]

    from robot_config import robots
    setup_robot(*robots[robot_id]())

    if cmd == 'jump':
        optimize_jump(toi=True, visualize=True)
    else:
        clear_states()
        forward()

    clear_states()
    forward('final{}'.format(robot_id))


if __name__ == '__main__':
    main()