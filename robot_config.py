import math

global objects, springs
objects = []
springs = []



def add_object(x, halfsize, rotation=0):
    objects.append([x, halfsize, rotation])
    return len(objects) - 1



def add_spring(a, b, offset_a, offset_b, length, stiffness, actuation=0.0):
    springs.append([a, b, offset_a, offset_b, length, stiffness, actuation])


def robotA():
    add_object(x=[0.3, 0.25], halfsize=[0.15, 0.03])
    add_object(x=[0.2, 0.15], halfsize=[0.03, 0.02])
    add_object(x=[0.3, 0.15], halfsize=[0.03, 0.02])
    add_object(x=[0.4, 0.15], halfsize=[0.03, 0.02])
    add_object(x=[0.4, 0.3], halfsize=[0.005, 0.03])

    l = 0.12
    s = 15
    add_spring(0, 1, [-0.03, 0.00], [0.0, 0.0], l, s)
    add_spring(0, 1, [-0.1, 0.00], [0.0, 0.0], l, s)
    add_spring(0, 2, [-0.03, 0.00], [0.0, 0.0], l, s)
    add_spring(0, 2, [0.03, 0.00], [0.0, 0.0], l, s)
    add_spring(0, 3, [0.03, 0.00], [0.0, 0.0], l, s)
    add_spring(0, 3, [0.1, 0.00], [0.0, 0.0], l, s)

    add_spring(0, 4, [0.1, 0], [0, -0.05], -1, s)

    return objects, springs, 0


def robotC():
    add_object(x=[0.3, 0.25], halfsize=[0.15, 0.03])
    add_object(x=[0.2, 0.15], halfsize=[0.03, 0.02])
    add_object(x=[0.3, 0.15], halfsize=[0.03, 0.02])
    add_object(x=[0.4, 0.15], halfsize=[0.03, 0.02])

    l = 0.12
    s = 15
    add_spring(0, 1, [-0.03, 0.00], [0.0, 0.0], l, s)
    add_spring(0, 1, [-0.1, 0.00], [0.0, 0.0], l, s)
    add_spring(0, 2, [-0.03, 0.00], [0.0, 0.0], l, s)
    add_spring(0, 2, [0.03, 0.00], [0.0, 0.0], l, s)
    add_spring(0, 3, [0.03, 0.00], [0.0, 0.0], l, s)
    add_spring(0, 3, [0.1, 0.00], [0.0, 0.0], l, s)

    return objects, springs, 3


l_thigh_init_ang = 10
l_calf_init_ang = -10
r_thigh_init_ang = 10
r_calf_init_ang = -10
initHeight = 0.15

hip_pos = [0.3, 0.5 + initHeight]
thigh_half_length = 0.11
calf_half_length = 0.11

foot_half_length = 0.08



def build_human_skeleton():
    global objects, springs
    objects = []
    springs = []

    """
    A 2D 'human' skeleton with a torso, head, and legs.
    (Arms have been removed to simplify the structure.)
    """

    torso_width = 0.04
    torso_height = 0.10
    leg_width = 0.03
    leg_height = 0.08
    head_width = 0.03
    head_height = 0.03

    torso_center = (0.5, 0.3)
    head_center = (0.5, 0.3 + torso_height/2 + head_height/2 + 0.05)

    # Leg centers
    left_thigh_center  = (0.5 - 0.02, 0.3 - torso_height/2 - leg_height/2)
    right_thigh_center = (0.5 + 0.02, 0.3 - torso_height/2 - leg_height/2)
    left_calf_center  = (left_thigh_center[0],  left_thigh_center[1] - 0.09)
    right_calf_center = (right_thigh_center[0], right_thigh_center[1] - 0.09)

    objects = [
        # 0: Torso
        [torso_center, (torso_width/2, torso_height/2), 0.0],
        # 1: Head
        [head_center, (head_width/2, head_height/2), 0.0],
        # 2: Left thigh
        [left_thigh_center, (leg_width/2, leg_height/2), 0.0],
        # 3: Left calf
        [left_calf_center, (leg_width/2, leg_height/2), 0.0],
        # 4: Right thigh
        [right_thigh_center, (leg_width/2, leg_height/2), 0.0],
        # 5: Right calf
        [right_calf_center, (leg_width/2, leg_height/2), 0.0],
    ]

    TORSO = 0
    HEAD = 1
    L_THIGH = 2
    L_CALF = 3
    R_THIGH = 4
    R_CALF = 5

    springs = []

    def short_spring(a, b, offA, offB, stiffness=100.0):
        return [a, b, offA, offB,
                0.05,         # rest length
                stiffness,
                0.0]          # no actuation

    def joint_spring(a, b, offA, offB):
        return [a, b, offA, offB,
                -1,
                0.0,
                0.0]

    def knee_spring(a, b, offA, offB, rest_len=0.12, stiffness=200.0, act=0.10):
        return [a, b, offA, offB,
                rest_len,  # rest length for knee extension
                stiffness,
                act]

    # Connect head to torso
    springs.append(short_spring(TORSO, HEAD,
        (0.0, torso_height/2),
        (0.0, -head_height/2)))

    # Hip joints: attach thighs rigidly to the torso
    springs.append(joint_spring(TORSO, L_THIGH,
        (-torso_width/2, -torso_height/2),
        (0.0, leg_height/2)))
    springs.append(joint_spring(TORSO, R_THIGH,
        (torso_width/2, -torso_height/2),
        (0.0, leg_height/2)))

    # Knee springs: these are our actuated piston joints
    springs.append(knee_spring(L_THIGH, L_CALF,
        (0.0, -leg_height/2),
        (0.0, leg_height/2),
        rest_len=0.10, stiffness=200.0, act=0.10))
    springs.append(knee_spring(R_THIGH, R_CALF,
        (0.0, -leg_height/2),
        (0.0, leg_height/2),
        rest_len=0.10, stiffness=200.0, act=0.10))

    head_id = HEAD
    return (objects, springs, head_id)




def rotAlong(half_length, deg, center):
    ang = math.radians(deg)
    return [
        half_length * math.sin(ang) + center[0],
        -half_length * math.cos(ang) + center[1]
    ]


half_hip_length = 0.08


def robotLeg():
    #hip
    add_object(hip_pos, halfsize=[0.06, half_hip_length])
    hip_end = [hip_pos[0], hip_pos[1] - (half_hip_length - 0.01)]

    #left
    l_thigh_center = rotAlong(thigh_half_length, l_thigh_init_ang, hip_end)
    l_thigh_end = rotAlong(thigh_half_length * 2.0, l_thigh_init_ang, hip_end)
    add_object(l_thigh_center,
               halfsize=[0.02, thigh_half_length],
               rotation=math.radians(l_thigh_init_ang))
    add_object(rotAlong(calf_half_length, l_calf_init_ang, l_thigh_end),
               halfsize=[0.02, calf_half_length],
               rotation=math.radians(l_calf_init_ang))
    l_calf_end = rotAlong(2.0 * calf_half_length, l_calf_init_ang, l_thigh_end)
    add_object([l_calf_end[0] + foot_half_length, l_calf_end[1]],
               halfsize=[foot_half_length, 0.02])

    #right
    add_object(rotAlong(thigh_half_length, r_thigh_init_ang, hip_end),
               halfsize=[0.02, thigh_half_length],
               rotation=math.radians(r_thigh_init_ang))
    r_thigh_end = rotAlong(thigh_half_length * 2.0, r_thigh_init_ang, hip_end)
    add_object(rotAlong(calf_half_length, r_calf_init_ang, r_thigh_end),
               halfsize=[0.02, calf_half_length],
               rotation=math.radians(r_calf_init_ang))
    r_calf_end = rotAlong(2.0 * calf_half_length, r_calf_init_ang, r_thigh_end)
    add_object([r_calf_end[0] + foot_half_length, r_calf_end[1]],
               halfsize=[foot_half_length, 0.02])

    s = 200

    thigh_relax = 0.9
    leg_relax = 0.9
    foot_relax = 0.7

    thigh_stiff = 5
    leg_stiff = 20
    foot_stiff = 40

    #left springs
    add_spring(0, 1, [0, (half_hip_length - 0.01) * 0.4],
               [0, -thigh_half_length],
               thigh_relax * (2.0 * thigh_half_length + 0.22), thigh_stiff)
    add_spring(1, 2, [0, thigh_half_length], [0, -thigh_half_length],
               leg_relax * 4.0 * thigh_half_length, leg_stiff, 0.08)
    add_spring(
        2, 3, [0, 0], [foot_half_length, 0],
        foot_relax *
        math.sqrt(pow(thigh_half_length, 2) + pow(2.0 * foot_half_length, 2)),
        foot_stiff)

    add_spring(0, 1, [0, -(half_hip_length - 0.01)], [0.0, thigh_half_length],
               -1, s)
    add_spring(1, 2, [0, -thigh_half_length], [0.0, thigh_half_length], -1, s)
    add_spring(2, 3, [0, -thigh_half_length], [-foot_half_length, 0], -1, s)

    #right springs
    add_spring(0, 4, [0, (half_hip_length - 0.01) * 0.4],
               [0, -thigh_half_length],
               thigh_relax * (2.0 * thigh_half_length + 0.22), thigh_stiff)
    add_spring(4, 5, [0, thigh_half_length], [0, -thigh_half_length],
               leg_relax * 4.0 * thigh_half_length, leg_stiff, 0.08)
    add_spring(
        5, 6, [0, 0], [foot_half_length, 0],
        foot_relax *
        math.sqrt(pow(thigh_half_length, 2) + pow(2.0 * foot_half_length, 2)),
        foot_stiff)

    add_spring(0, 4, [0, -(half_hip_length - 0.01)], [0.0, thigh_half_length],
               -1, s)
    add_spring(4, 5, [0, -thigh_half_length], [0.0, thigh_half_length], -1, s)
    add_spring(5, 6, [0, -thigh_half_length], [-foot_half_length, 0], -1, s)

    return objects, springs, 3


def robotB():
    body = add_object([0.15, 0.25], [0.1, 0.03])
    back = add_object([0.08, 0.22], [0.03, 0.10])
    front = add_object([0.22, 0.22], [0.03, 0.10])

    rest_length = 0.22
    stiffness = 50
    act = 0.03
    add_spring(body,
               back, [0.08, 0.02], [0.0, -0.08],
               rest_length,
               stiffness,
               actuation=act)
    add_spring(body,
               front, [-0.08, 0.02], [0.0, -0.08],
               rest_length,
               stiffness,
               actuation=act)

    add_spring(body, back, [-0.08, 0.0], [0.0, 0.03], -1, stiffness)
    add_spring(body, front, [0.08, 0.0], [0.0, 0.03], -1, stiffness)

    return objects, springs, body

def human_skeleton_robot():
    return build_human_skeleton()


robots = [robotA, robotB, robotLeg, human_skeleton_robot]