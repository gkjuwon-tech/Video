"""CAM-04 테스트 공용 도우미. 구현 내부가 아니라 계약서(CAM-04.md)의 공식만 쓴다.

세계 기준 배치: C_0 = 단위행렬(원점, +z를 봄), 얼굴 중심은 (0, 0, D). 세계 up = (0, -1, 0)(OpenCV y 아래).
얼굴 좌표계(계약서): right = up x front, 카메라 상대 벡터 v = c - center, x = v.right, y = v.up, z = v.front,
yaw = atan2(x, z), pitch = atan2(y, hypot(x, z)).
"""
import math

import numpy as np

UP = np.array([0.0, -1.0, 0.0])
CAP_HALF_DEG = 45.0          # 계약서 상수 FACE_CAP_HALF_DEG
HEAD_R = 0.10                # 계약서 상수 HEAD_RADIUS_M


def fe():
    from buc.camera import face_envelope
    return face_envelope


def make_face(front=(0.0, 0.0, 1.0), d=2.0, h_f=0.1, sigma_z=0.0, **kw):
    """얼굴 중심 (0,0,d). front 기본 (0,0,1) = 카메라를 등진 뒷모습. 정면(카메라를 봄)은 front=(0,0,-1)."""
    return fe().FaceSpec(center=[0.0, 0.0, d], front=list(front), up=list(UP), h_f=h_f, sigma_z=sigma_z, **kw)


def frame(face):
    """계약서 정의의 얼굴 좌표 기저 (right, up, front). 구현의 속성이 아니라 공식으로 다시 만든다."""
    f = np.asarray(face.front, float)
    f = f / np.linalg.norm(f)
    u = np.asarray(face.up, float)
    u = u - f * (f @ u)
    u = u / np.linalg.norm(u)
    return np.cross(u, f), u, f


def cam_pos(face, yaw_deg, pitch_deg=0.0, dist=2.0):
    """얼굴 좌표계 (yaw, pitch, dist) 에서 세계 위치."""
    r, u, f = frame(face)
    c = np.asarray(face.center, float)
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    return c + dist * (math.cos(p) * (math.sin(y) * r + math.cos(y) * f) + math.sin(p) * u)


def look_at(pos, target, down=UP * -1.0):
    """OpenCV c2w (z = 타깃 방향, y 는 대략 아래)."""
    z = np.asarray(target, float) - np.asarray(pos, float)
    z = z / np.linalg.norm(z)
    x = np.cross(down, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    out = np.eye(4)
    out[:3, 0], out[:3, 1], out[:3, 2], out[:3, 3] = x, y, z, pos
    return out


def pose_at(face, yaw_deg, pitch_deg=0.0, dist=2.0):
    return look_at(cam_pos(face, yaw_deg, pitch_deg, dist), face.center)


def identity_c0():
    return np.eye(4)


def orbit_path(face, c0_pos, delta_yaw_deg, delta_pitch_deg=0.0, n=61):
    """관측 카메라 위치에서 얼굴 중심 둘레로 yaw 를 0 → delta 로 부드럽게 돌린 c2w (n,4,4). 거리 일정."""
    r, u, f = frame(face)
    v = np.asarray(c0_pos, float) - np.asarray(face.center, float)
    d0 = np.linalg.norm(v)
    y0 = math.degrees(math.atan2(v @ r, v @ f))
    p0 = math.degrees(math.atan2(v @ u, math.hypot(v @ r, v @ f)))
    ts = np.linspace(0.0, 1.0, n)
    return np.stack([pose_at(face, y0 + t * delta_yaw_deg, p0 + t * delta_pitch_deg, d0) for t in ts])


def push_path(face, c0_pos, m, n=41):
    """얼굴 중심 쪽 직선 전진: 마지막 거리 = d0 / m (m<1 은 후퇴)."""
    c0 = np.asarray(c0_pos, float)
    ctr = np.asarray(face.center, float)
    out = []
    for t in np.linspace(0.0, 1.0, n):
        p = c0 + (ctr - c0) * (1.0 - 1.0 / m) * t
        out.append(look_at(p, ctr))
    return np.stack(out)


def ref_exposure(face, p0, p, nt=400, nph=800):
    """계약서 정의의 독립 구현(위도-경도 등면적 구적). 머리 구 위 얼굴 캡(반각 45도)에서
    p 에서는 보이고 p0 에서는 안 보이는 면적 비율. 가시: n . (cam - center) > r_head."""
    r, u, f = frame(face)
    ct = 1.0 - (1.0 - math.cos(math.radians(CAP_HALF_DEG))) * (np.arange(nt) + 0.5) / nt
    ph = (np.arange(nph) + 0.5) * 2.0 * math.pi / nph
    CT, PH = np.meshgrid(ct, ph, indexing="ij")
    ST = np.sqrt(1.0 - CT ** 2)
    n = np.stack([ST * np.cos(PH), ST * np.sin(PH), CT], -1).reshape(-1, 3)   # (right, up, front)

    def loc(q):
        v = np.asarray(q, float) - np.asarray(face.center, float)
        return np.array([v @ r, v @ u, v @ f])

    rh = getattr(face, "head_radius_m", HEAD_R)
    seen0 = n @ loc(p0) > rh
    seen = n @ loc(p) > rh
    return float(np.mean(seen & ~seen0))


def solve_angle_for_exposure(face, p0_yaw, target, toward_sign, dist=2.0, lo=0.0, hi=120.0):
    """ref_exposure 가 target 이 되는 yaw 이동량(도)을 이분법으로 찾는다(단조 구간 가정)."""
    p0 = cam_pos(face, p0_yaw, 0.0, dist)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        e = ref_exposure(face, p0, cam_pos(face, p0_yaw + toward_sign * mid, 0.0, dist), 200, 400)
        lo, hi = (mid, hi) if e < target else (lo, mid)
    return 0.5 * (lo + hi)


def random_rigid(seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q, rng.uniform(-3, 3, size=3)


def transform_poses(c2w, R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return np.einsum("ij,njk->nik", T, c2w)


def transform_face(face, R, t):
    return fe().FaceSpec(center=R @ np.asarray(face.center) + t, front=R @ np.asarray(face.front),
                         up=R @ np.asarray(face.up), h_f=face.h_f, sigma_z=face.sigma_z,
                         head_radius_m=face.head_radius_m)
