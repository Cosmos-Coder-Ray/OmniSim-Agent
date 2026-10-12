import json
from pathlib import Path

import pybullet as p
import pybullet_data

ROOT = Path(__file__).resolve().parents[1]


def dump_urdf_limits(output: Path = ROOT / "data" / "urdf_limits.json") -> None:
    client_id = p.connect(p.DIRECT)
    try:
        robot = p.loadURDF(
            str(Path(pybullet_data.getDataPath()) / "franka_panda" / "panda.urdf"),
            useFixedBase=True,
            physicsClientId=client_id,
        )
        rows = []
        for joint in range(7):
            info = p.getJointInfo(robot, joint, physicsClientId=client_id)
            rows.append(
                {
                    "joint": joint,
                    "effort": info[10],
                    "velocity": info[11],
                    "lower": info[8],
                    "upper": info[9],
                }
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        p.disconnect(physicsClientId=client_id)


if __name__ == "__main__":
    dump_urdf_limits()
