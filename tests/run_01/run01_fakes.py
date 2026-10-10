"""RUN-01 테스트용 가짜 Vast API·주입 시계. 네트워크 없이 VastAPI 이음매를 흉내 낸다."""
import calendar
import copy

from buc.rt import vast as V

# 2026-10-10T00:00:00Z (UTC 자정)
DAY0 = float(calendar.timegm((2026, 10, 10, 0, 0, 0)))
H = 3600.0


class Clock:
    def __init__(self, t: float = DAY0 + 10 * H):
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def set(self, t: float) -> None:
        self.t = float(t)


def raw_offer(id=1, machine_id=None, dph=0.40, gpu="RTX 5090", num_gpus=1, rel=0.995, geo="Seoul, KR",
              cpu=16, ram_mb=64 * 1024, disk=200, inet=900, cuda=12.8, verified=True, rentable=True):
    return {
        "id": id, "machine_id": id * 10 if machine_id is None else machine_id, "gpu_name": gpu,
        "num_gpus": num_gpus, "dph_total": dph, "reliability2": rel, "geolocation": geo,
        "cpu_cores_effective": cpu, "cpu_ram": ram_mb, "disk_space": disk, "inet_down": inet,
        "cuda_max_good": cuda, "verification": "verified" if verified else "unverified", "rentable": rentable,
    }


class FakeVast:
    """VastAPI 프로토콜 구현. calls에 (메서드, 인자) 기록. fail에 (메서드, iid) 넣으면 VastAPIError."""

    def __init__(self, offers=None, clock=None):
        self.offers = list(offers or [])
        self.instances = {}
        self.calls = []
        self.fail = set()
        self.clock = clock or Clock()
        self.next_id = 5000

    def _maybe_fail(self, name, iid):
        if (name, iid) in self.fail:
            raise V.VastAPIError("fake failure %s %s" % (name, iid))

    def search_offers(self, query):
        self.calls.append(("search_offers", copy.deepcopy(query)))
        return [dict(o) for o in self.offers]

    def create_instance(self, offer_id, config):
        self.calls.append(("create_instance", offer_id, copy.deepcopy(config)))
        o = next(x for x in self.offers if x["id"] == offer_id)
        self.offers.remove(o)  # 빌린 매물은 시장에서 사라진다
        iid = self.next_id
        self.next_id += 1
        self.instances[iid] = {
            "id": iid, "machine_id": o["machine_id"], "label": config["label"], "gpu_name": o["gpu_name"],
            "num_gpus": o["num_gpus"], "dph_total": o["dph_total"], "actual_status": "loading",
            "intended_status": "running", "start_date": self.clock(),
        }
        return iid

    def show_instance(self, iid):
        self.calls.append(("show_instance", iid))
        if iid not in self.instances:
            raise V.VastNotFound(str(iid))
        return dict(self.instances[iid])

    def list_instances(self):
        self.calls.append(("list_instances",))
        return [dict(v) for v in self.instances.values()]

    def set_state(self, iid, state):
        self.calls.append(("set_state", iid, state))
        self._maybe_fail("set_state", iid)
        if iid not in self.instances:
            raise V.VastNotFound(str(iid))
        d = self.instances[iid]
        d["intended_status"] = state
        d["actual_status"] = "running" if state == "running" else "exited"

    def destroy_instance(self, iid):
        self.calls.append(("destroy_instance", iid))
        self._maybe_fail("destroy_instance", iid)
        if iid not in self.instances:
            raise V.VastNotFound(str(iid))
        del self.instances[iid]

    # 도우미
    def n(self, name):
        return sum(1 for c in self.calls if c[0] == name)

    def set_status(self, iid, actual=None, intended=None):
        if actual is not None:
            self.instances[iid]["actual_status"] = actual
        if intended is not None:
            self.instances[iid]["intended_status"] = intended
