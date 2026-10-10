"""DAT-16 도메인 앵커. 테스트 작성자가 독립적으로 정한 '반드시 있어야 하는 클래스'와 그 합리 범위.
각 항목: 클래스 ID -> (dim, 중앙값 하한 m, 중앙값 상한 m, σ_log 하한, σ_log 상한, group).
범위는 일부러 넉넉하다(표준 규격이 있으면 좁고, 변동이 크면 넓다). 숫자 수정은 도메인 검수자의 몫이다.
모든 앵커는 tier 1이어야 한다."""

ANCHORS = {
    # 건축
    "door_interior": ("height", 1.98, 2.12, 0.02, 0.06, "architecture"),
    "door_exterior": ("height", 1.98, 2.15, 0.02, 0.07, "architecture"),
    "stair_riser": ("height", 0.155, 0.185, 0.03, 0.12, "architecture"),
    "stair_tread": ("depth", 0.24, 0.32, 0.04, 0.15, "architecture"),
    "room_ceiling": ("height", 2.4, 2.8, 0.03, 0.12, "architecture"),
    "building_story": ("height", 2.7, 3.6, 0.04, 0.15, "architecture"),
    "window_residential": ("height", 0.9, 1.6, 0.10, 0.40, "architecture"),
    "handrail": ("height", 0.85, 1.0, 0.02, 0.10, "architecture"),
    "light_switch": ("height", 0.10, 0.13, 0.02, 0.12, "architecture"),
    "house": ("height", 4.5, 9.0, 0.15, 0.60, "architecture"),
    # 가구
    "chair_seat": ("height", 0.42, 0.48, 0.03, 0.10, "furniture"),
    "chair": ("height", 0.78, 1.0, 0.05, 0.25, "furniture"),
    "dining_table": ("height", 0.72, 0.78, 0.02, 0.07, "furniture"),
    "desk": ("height", 0.70, 0.78, 0.02, 0.08, "furniture"),
    "coffee_table": ("height", 0.38, 0.50, 0.05, 0.20, "furniture"),
    "sofa": ("length", 1.8, 2.2, 0.08, 0.30, "furniture"),
    "bed_double": ("length", 1.95, 2.15, 0.02, 0.08, "furniture"),
    "bookshelf": ("height", 1.4, 2.2, 0.10, 0.40, "furniture"),
    "kitchen_counter": ("height", 0.85, 0.95, 0.02, 0.08, "furniture"),
    "bathtub": ("length", 1.5, 1.8, 0.04, 0.15, "furniture"),
    # 가전
    "refrigerator": ("height", 1.5, 1.9, 0.06, 0.20, "appliance"),
    "washing_machine": ("height", 0.8, 0.9, 0.02, 0.08, "appliance"),
    # 사람
    "person_adult": ("height", 1.62, 1.76, 0.04, 0.09, "person"),
    "person_child": ("height", 1.0, 1.35, 0.10, 0.30, "person"),
    "hand_adult": ("length", 0.17, 0.20, 0.03, 0.10, "person"),
    "head_adult": ("height", 0.21, 0.26, 0.03, 0.10, "person"),
    # 동물
    "dog": ("height", 0.35, 0.65, 0.20, 0.60, "animal"),
    "cat": ("height", 0.20, 0.32, 0.10, 0.30, "animal"),
    "horse": ("height", 1.4, 1.7, 0.05, 0.15, "animal"),
    "cow": ("height", 1.2, 1.5, 0.05, 0.15, "animal"),
    "elephant_african": ("height", 2.8, 3.5, 0.05, 0.15, "animal"),
    # 탈것
    "car": ("height", 1.4, 1.6, 0.03, 0.15, "vehicle"),
    "bus": ("height", 2.9, 3.5, 0.04, 0.15, "vehicle"),
    "bicycle": ("length", 1.6, 1.85, 0.03, 0.12, "vehicle"),
    "motorcycle": ("length", 1.9, 2.4, 0.05, 0.18, "vehicle"),
    # 거리
    "stop_sign": ("width", 0.6, 0.9, 0.03, 0.15, "street"),
    "fire_hydrant": ("height", 0.55, 0.9, 0.08, 0.25, "street"),
    "street_lamp": ("height", 4.0, 10.0, 0.15, 0.50, "street"),
    "bench": ("length", 1.4, 2.0, 0.08, 0.30, "street"),
    "traffic_cone": ("height", 0.45, 0.9, 0.10, 0.30, "street"),
    "manhole_cover": ("diameter", 0.55, 0.8, 0.04, 0.15, "street"),
    # 식기·용기
    "coffee_mug": ("height", 0.08, 0.11, 0.04, 0.15, "tableware"),
    "wine_bottle": ("height", 0.29, 0.33, 0.02, 0.08, "container"),
    "beer_bottle": ("height", 0.20, 0.26, 0.03, 0.10, "container"),
    "soda_can": ("height", 0.115, 0.13, 0.02, 0.05, "container"),
    "water_bottle": ("height", 0.19, 0.25, 0.08, 0.30, "container"),
    "dinner_plate": ("diameter", 0.24, 0.29, 0.03, 0.12, "tableware"),
    "wine_glass": ("height", 0.16, 0.24, 0.08, 0.25, "tableware"),
    "fork": ("length", 0.17, 0.21, 0.03, 0.12, "tableware"),
    # 전자
    "laptop": ("width", 0.30, 0.37, 0.05, 0.15, "electronics"),
    "smartphone": ("height", 0.14, 0.165, 0.03, 0.10, "electronics"),
    "television": ("width", 0.9, 1.3, 0.15, 0.40, "electronics"),
    "computer_monitor": ("width", 0.45, 0.65, 0.10, 0.30, "electronics"),
    "keyboard": ("width", 0.36, 0.47, 0.04, 0.15, "electronics"),
    "computer_mouse": ("length", 0.095, 0.125, 0.04, 0.15, "electronics"),
    # 소품·도구
    "ballpoint_pen": ("length", 0.13, 0.15, 0.02, 0.10, "tool"),
    "sheet_paper": ("height", 0.275, 0.30, 0.02, 0.06, "tool"),
    "book": ("height", 0.19, 0.26, 0.08, 0.30, "tool"),
    "credit_card": ("width", 0.0850, 0.0862, 0.02, 0.03, "tool"),
    "backpack": ("height", 0.40, 0.55, 0.08, 0.25, "clothing"),
    "shoe_adult": ("length", 0.26, 0.30, 0.04, 0.12, "clothing"),
    # 스포츠
    "basketball": ("diameter", 0.235, 0.25, 0.02, 0.05, "sport"),
    "soccer_ball": ("diameter", 0.215, 0.225, 0.02, 0.05, "sport"),
    "tennis_ball": ("diameter", 0.065, 0.070, 0.02, 0.05, "sport"),
    "golf_ball": ("diameter", 0.0420, 0.0435, 0.02, 0.03, "sport"),
    "tennis_racket": ("length", 0.66, 0.70, 0.02, 0.08, "sport"),
    # 음식
    "egg_chicken": ("length", 0.052, 0.062, 0.03, 0.10, "food"),
    "apple": ("diameter", 0.065, 0.085, 0.05, 0.15, "food"),
    "banana": ("length", 0.17, 0.21, 0.06, 0.18, "food"),
    # 식물
    "tree": ("height", 8.0, 25.0, 0.25, 0.70, "plant"),
}

# 중앙값 기준 엄격 증가 사슬(크기 순서 상식). 연속 항 사이에 최소 비율을 요구한다.
CHAIN = [
    "golf_ball", "egg_chicken", "tennis_ball", "coffee_mug", "soda_can", "smartphone", "wine_bottle",
    "chair_seat", "dining_table", "person_adult", "door_interior", "building_story", "house", "tree",
]
CHAIN_MIN_RATIO = 1.03

# 별칭 계약: 일반어 -> 정본 ID (이 별칭들은 반드시 해석돼야 한다).
REQUIRED_ALIASES = {"person": "person_adult", "human": "person_adult", "door": "door_interior"}
# 정본 ID 자체로 해석돼야 하는 일반 클래스.
REQUIRED_CANONICAL = ["chair", "car", "tree", "dog", "cat", "bicycle", "laptop", "bench", "book"]
