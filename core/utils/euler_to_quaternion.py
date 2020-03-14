import numpy as np
import math



def rotation_mat_to_quaterion(rotation_matrix):
    m11 = rotation_matrix[0, 0]
    m12 = rotation_matrix[0, 1]
    m13 = rotation_matrix[0, 2]
    m21 = rotation_matrix[1, 0]
    m22 = rotation_matrix[1, 1]
    m23 = rotation_matrix[1, 2]
    m31 = rotation_matrix[2, 0]
    m32 = rotation_matrix[2, 1]
    m33 = rotation_matrix[2, 2]

    w_raw = np.sqrt(abs(m11 + m22 + m33 + 1)) / 2
    x_raw = np.sqrt(abs(m11 - m22 - m33 + 1)) / 2
    y_raw = np.sqrt(abs(-m11 + m22 - m33 + 1)) / 2
    z_raw = np.sqrt(abs(-m11 - m22 + m33 + 1)) / 2

    q = np.argmax([w_raw**2, x_raw**2, y_raw**2, z_raw**2])
    if q == 0:
        w = w_raw
        x = (m23 - m32) / (4 * abs(w))
        y = (m31 - m13) / (4 * abs(w))
        z = (m12 - m21) / (4 * abs(w))
    elif q == 1:
        x = x_raw
        w = (m23 - m32) / (4 * abs(x))
        y = (m12 + m21) / (4 * abs(x))
        z = (m31 + m13) / (4 * abs(x))
    elif q == 2:
        y = y_raw
        w = (m31 - m13) / (4 * abs(y))
        x = (m12 + m21) / (4 * abs(y))
        z = (m23 + m32) / (4 * abs(y))
    else:
        z = z_raw
        w = (m12 - m21) / (4 * abs(z))
        x = (m31 + m13) / (4 * abs(z))
        y = (m23 + m32) / (4 * abs(z))

    return w, x, y, z



def quaternion_to_rotation_mat(w, x, y, z):
    m11 = 1 - 2*y*y - 2*z*z
    m12 = 2*x*y + 2*w*z
    m13 = 2*x*z - 2*w*y
    m21 = 2*x*y - 2*w*z
    m22 = 1- 2*x*x - 2*z*z
    m23 = 2*y*z + 2*w*x
    m31 = 2*x*z + 2*w*y
    m32 = 2*y*z - 2*w*x
    m33 = 1 - 2*x*x - 2*y*y

    rotation_matrix = np.matrix([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])
    return rotation_matrix


def euler_to_rotation_mat(x, y, z):
    R_x = np.matrix([[1,         0,                  0                   ],
                    [0,         math.cos(x), -math.sin(x) ],
                    [0,         math.sin(x), math.cos(x)  ]
                    ])

    R_y = np.matrix([[math.cos(y),    0,      math.sin(y)  ],
                    [0,                     1,      0                   ],
                    [-math.sin(y),   0,      math.cos(y)  ]
                    ])

    R_z = np.matrix([[math.cos(z),    -math.sin(z),    0],
                    [math.sin(z),    math.cos(z),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def rotation_mat_to_euler(R):
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if not singular :
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])
