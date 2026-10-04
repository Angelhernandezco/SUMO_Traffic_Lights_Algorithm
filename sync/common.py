"""Shared resources and fixed experimental configuration."""
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMO_HOME = Path(os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo"))
if (SUMO_HOME / "tools").is_dir():
    sys.path.append(str(SUMO_HOME / "tools"))
import traci
from traci import constants as tc

NET = ROOT / "maps/master_slave.net.xml"
CHECKPOINT = ROOT / "policy/models/model_future_v39_yellow_test36_3.pth"
DEMAND = ROOT / "sync/demand"
TLS = ("J0", "J2", "J10", "J16")
RECEIVERS = {"J2": (2, "E1", "E5"), "J10": (4, "E5", "E10"), "J16": (2, "E10", "E13")}
OFFSETS = (0, 0, 72, 24)
GREEN_MIN, GREEN_MAX, YELLOW = 5, 45, 4
CORRIDOR_PHASE, FLIGHT_TIME, END = 2, 12, 3600

def sumo_binary(gui=False):
    name = "sumo-gui" if gui else "sumo"
    found = shutil.which(name)
    if found:
        return found
    binary = SUMO_HOME / "bin" / (name + (".exe" if os.name == "nt" else ""))
    if not binary.is_file():
        raise FileNotFoundError("Set SUMO_HOME or add SUMO bin to PATH")
    return str(binary)

def static_net(path, offsets=OFFSETS):
    """Validate the base network and change only offsets in a temporary copy."""
    xml = NET.read_text(encoding="utf-8")
    root = ET.fromstring(xml)
    signatures = []
    for tls in TLS:
        logics = root.findall(f"tlLogic[@id='{tls}']")
        if len(logics) != 1 or logics[0].get("programID") != "0":
            raise RuntimeError(f"Unexpected programs for {tls}")
        signatures.append([(float(p.get("duration")), p.get("state")) for p in logics[0]])
    if any(s != signatures[0] for s in signatures) or [d for d, _ in signatures[0]] != [15,4]*4:
        raise RuntimeError("Base network must have identical 76 s programs")
    for tls, offset in zip(TLS, offsets):
        pattern = rf'(<tlLogic id="{tls}" type="static" programID="0" offset=")\d+("|>)'
        xml, count = re.subn(pattern, lambda m: m.group(1) + str(offset % 76) + m.group(2), xml)
        if count != 1:
            raise RuntimeError(f"Missing offset for {tls}")
    path.write_text(xml, encoding="utf-8")
