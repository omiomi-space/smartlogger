"""SmartLogger 3D viewer.

Receives the JSON/UDP stream from the SmartLogger Android app and shows the phone as a
3D model in real time: orientation from the rotation-vector quaternion, horizontal
position from a Kalman filter (fusion.py) that dead-reckons with the world-frame linear
acceleration between GPS fixes, corrected by GPS position/velocity, barometric height and
zero-velocity updates when the phone is still.

World frame = Android world frame = ENU:  X = East, Y = North, Z = Up.

Controls
  left-drag : orbit camera        wheel : zoom
  F         : toggle follow phone  R     : reset origin / trail
  T         : toggle trail          A     : toggle accel vector
  K         : EKF <-> GPS-only       G     : toggle raw GPS fix markers
  Esc       : quit
"""

import argparse
import json
import math
import socket
import subprocess
import sys
import threading
import time
from collections import deque

import numpy as np

from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from discovery import Discovery
from fusion import Fusion
from panda3d.core import (
    AmbientLight, DirectionalLight, GeomNode, LineSegs, NodePath, Quat, TextNode,
    Vec3, Vec4, loadPrcFileData, CardMaker, AntialiasAttrib,
)



# --------------------------------------------------------------------------- receiver

class Receiver(threading.Thread):
    """Background UDP listener that keeps the latest packet and a packet-rate estimate."""

    HELLO = b"SMARTLOGGER_HELLO"

    def __init__(self, port: int, phone: str = None, alt_source: str = "baro", phone_port: int = None):
        super().__init__(daemon=True)
        self.fusion = Fusion()
        self.alt_source = alt_source
        self._new_fix_points = []
        self.record = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            # Windows would otherwise let a second viewer share the port and steal half the packets
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(0.5)
        # how to find the phone: "auto" (network discovery), "adb", "none" or a fixed IP[:port]
        phone = phone or "auto"
        self.phone_mode = phone if phone in ("auto", "adb", "none") else "ip"
        self.phone_auto = self.phone_mode == "adb"
        self.phone = None
        self.port = phone_port or port        # the app's UDP port (hello destination)
        self.discovery = Discovery()
        self.discovery_status = ""
        if self.phone_mode == "ip":
            host, _, pport = phone.partition(":")
            self.phone = (host, int(pport) if pport else self.port)
        self.lock = threading.Lock()
        self.latest = None
        self.sender = None
        self.last_rx = 0.0
        self.count = 0
        self.lost = 0
        self._last_seq = None
        self._stamps = deque(maxlen=200)
        self.new_gps = deque(maxlen=50)
        self._last_gps_t = None

    def _hello_loop(self):
        """Periodically say hello to the phone from the receive socket.

        The phone replies to this address/port, so the data is a reply to a PC-initiated
        flow and passes stateful firewalls (ESET etc.) that drop unsolicited inbound UDP.
        """
        next_lookup = 0.0
        next_discovery = 0.0
        while True:
            now = time.time()
            quiet = now - self.last_rx > 3.0
            # network discovery: while nothing arrives, hello broadcast / ARP neighbours / sweep
            if self.phone_mode == "auto" and quiet and now >= next_discovery:
                for ip in self.discovery.targets():
                    try:
                        self.sock.sendto(self.HELLO, (ip, self.port))
                    except OSError:
                        pass
                self.discovery_status = self.discovery.status
                next_discovery = now + 2.0
            # with --phone adb: look the IP up again if we have none or the stream went quiet
            if self.phone_auto and now >= next_lookup and (self.phone is None or now - self.last_rx > 5.0):
                ip = phone_ip_from_adb()
                if ip:
                    if self.phone is None or self.phone[0] != ip:
                        print(f"[adb] phone IP: {ip}")
                    self.phone = (ip, self.port)
                next_lookup = now + 5.0
            if self.phone:
                try:
                    self.sock.sendto(self.HELLO, self.phone)
                except OSError:
                    pass
            time.sleep(1.0)

    def run(self):
        if self.phone_mode != "none":
            threading.Thread(target=self._hello_loop, daemon=True).start()
        while True:
            try:
                data, addr = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except ConnectionResetError:
                # Windows reports ICMP "port unreachable" (app not started yet) as a reset; ignore it.
                continue
            except OSError:
                return
            try:
                text = data.decode("utf-8")
                pkt = json.loads(text)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if self.record is not None:
                self.record.write(text + "\n")
            now = time.time()
            with self.lock:
                seq = pkt.get("seq")
                if seq is not None and self._last_seq is not None and seq > self._last_seq + 1:
                    self.lost += seq - self._last_seq - 1
                if seq is not None and (self._last_seq is None or seq > self._last_seq or seq < 5):
                    self._last_seq = seq
                self.latest = pkt
                self.sender = addr
                if self.phone_mode == "auto" and self.phone != addr:
                    self.phone = addr          # lock onto the device that answered
                    self.discovery_status = ""
                    print(f"[discovery] phone found at {addr[0]}:{addr[1]}")
                self.last_rx = now
                self.count += 1
                self._stamps.append(now)
                gps = pkt.get("gps")
                if gps and gps.get("t") != self._last_gps_t:
                    self._last_gps_t = gps.get("t")
                    self.new_gps.append(gps)
                fpkt = pkt if self.alt_source != "gps" else {k: v for k, v in pkt.items() if k != "p"}
                n_pts = len(self.fusion.gps_points)
                self.fusion.process(fpkt)
                if len(self.fusion.gps_points) != n_pts and self.fusion.gps_points:
                    self._new_fix_points.append(self.fusion.gps_points[-1])

    def reset_fusion(self):
        with self.lock:
            self.fusion.reset()
            self._new_fix_points.clear()

    def fusion_state(self):
        """Copy of the filter outputs; the GPS correction jump and new fix points are consumed."""
        with self.lock:
            f = self.fusion
            st = {"pos": f.pos.copy(), "vel": f.vel.copy(), "bias": f.bias.copy(), "jump": f.jump.copy(),
                  "still": f.still, "gps_age": f.gps_age(), "gps_interval": f.gps_interval,
                  "new_fixes": list(self._new_fix_points),
                  "steps": f.steps, "step_len": f.step_len, "walking": f.walking, "heading": f.heading,
                  "ar_state": f.ar_state, "ar_tracking": f.ar_tracking,
                  "pos_sigma": float(np.sqrt(max(f.P[0, 0], f.P[1, 1])))}
            f.jump[:] = 0
            self._new_fix_points.clear()
            return st

    def snapshot(self):
        with self.lock:
            stamps = list(self._stamps)
            gps = list(self.new_gps)
            self.new_gps.clear()
            rate = 0.0
            if len(stamps) > 1 and time.time() - stamps[-1] < 1.0:
                rate = (len(stamps) - 1) / max(stamps[-1] - stamps[0], 1e-6)
            return self.latest, self.sender, self.last_rx, rate, gps


# --------------------------------------------------------------------------- geometry helpers

def make_box(sx, sy, sz, color):
    """Axis-aligned box centred at origin built from 6 cards."""
    root = NodePath("box")
    cm = CardMaker("face")
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    faces = [
        # (frame args, pos, hpr)
        ((-hx, hx, -hy, hy), (0, 0, hz), (0, -90, 0)),   # top (+Z)
        ((-hx, hx, -hy, hy), (0, 0, -hz), (0, 90, 0)),   # bottom (-Z)
        ((-hx, hx, -hz, hz), (0, -hy, 0), (0, 0, 0)),    # -Y
        ((-hx, hx, -hz, hz), (0, hy, 0), (180, 0, 0)),   # +Y
        ((-hy, hy, -hz, hz), (hx, 0, 0), (90, 0, 0)),    # +X
        ((-hy, hy, -hz, hz), (-hx, 0, 0), (-90, 0, 0)),  # -X
    ]
    for frame, pos, hpr in faces:
        cm.setFrame(*frame)
        f = root.attachNewNode(cm.generate())
        f.setPos(*pos)
        f.setHpr(*hpr)
    root.setColor(*color)
    root.setTwoSided(True)
    return root


def make_lines(segments, thickness=2.0):
    ls = LineSegs()
    ls.setThickness(thickness)
    for (a, b, color) in segments:
        ls.setColor(*color)
        ls.moveTo(*a)
        ls.drawTo(*b)
    return NodePath(ls.create())


def build_phone(scale):
    """Phone model in Android device frame: X right, Y up the screen, Z out of the screen."""
    w, h, d = 0.075 * scale, 0.155 * scale, 0.009 * scale
    phone = NodePath("phone")
    body = make_box(w, h, d, (0.25, 0.27, 0.32, 1))
    body.reparentTo(phone)
    screen = make_box(w * 0.92, h * 0.9, d * 0.05, (0.05, 0.25, 0.55, 1))
    screen.setPos(0, -h * 0.01, d / 2 + d * 0.03)
    screen.setLightOff()
    screen.reparentTo(phone)
    # top speaker bar -> shows which way is "up" on the screen
    speaker = make_box(w * 0.3, h * 0.025, d * 0.1, (0.9, 0.9, 0.9, 1))
    speaker.setPos(0, h * 0.475, d / 2 + d * 0.06)
    speaker.setLightOff()
    speaker.reparentTo(phone)
    # camera bump on the back, top-left as seen from the back
    cam = make_box(w * 0.35, w * 0.35, d * 0.4, (0.1, 0.1, 0.1, 1))
    cam.setPos(-w * 0.25, h * 0.36, -d / 2 - d * 0.2)
    cam.reparentTo(phone)
    # device axes
    L = h * 0.75
    axes = make_lines([
        ((0, 0, 0), (L, 0, 0), (1, 0.2, 0.2, 1)),
        ((0, 0, 0), (0, L, 0), (0.2, 1, 0.2, 1)),
        ((0, 0, 0), (0, 0, L), (0.3, 0.5, 1, 1)),
    ], thickness=3)
    axes.setLightOff()
    axes.setDepthTest(False)
    axes.setBin("fixed", 10)
    axes.reparentTo(phone)
    return phone


def build_grid(size=200, step=1.0, major=10):
    ls = LineSegs()
    ls.setThickness(1)
    n = int(size / step)
    for i in range(-n, n + 1):
        v = i * step
        c = (0.35, 0.35, 0.4, 1) if i % major == 0 else (0.2, 0.2, 0.24, 1)
        ls.setColor(*c)
        ls.moveTo(v, -size, 0)
        ls.drawTo(v, size, 0)
        ls.moveTo(-size, v, 0)
        ls.drawTo(size, v, 0)
    np = NodePath(ls.create())
    np.setLightOff()
    return np


# --------------------------------------------------------------------------- app

class Viewer(ShowBase):
    def __init__(self, args):
        loadPrcFileData("", "window-title SmartLogger 3D Viewer\nwin-size 1280 800\nframebuffer-multisample 1\nmultisamples 4\nsync-video 1")
        super().__init__()
        self.args = args
        self.rx = Receiver(args.port, args.phone, args.alt, getattr(args, "phone_port", None))
        if getattr(args, "record", None):
            self.rx.record = open(args.record, "a", encoding="utf-8", buffering=1)
        self.rx.start()

        self.setBackgroundColor(0.08, 0.08, 0.1, 1)
        self.disableMouse()
        self.render.setAntialias(AntialiasAttrib.MAuto)

        amb = AmbientLight("amb")
        amb.setColor(Vec4(0.45, 0.45, 0.5, 1))
        self.render.setLight(self.render.attachNewNode(amb))
        sun = DirectionalLight("sun")
        sun.setColor(Vec4(0.8, 0.8, 0.75, 1))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(30, -60, 0)
        self.render.setLight(sun_np)

        build_grid(size=args.grid, step=1.0).reparentTo(self.render)
        world_axes = make_lines([
            ((0, 0, 0), (3, 0, 0), (1, 0.3, 0.3, 1)),
            ((0, 0, 0), (0, 3, 0), (0.3, 1, 0.3, 1)),
            ((0, 0, 0), (0, 0, 3), (0.4, 0.6, 1, 1)),
        ], thickness=3)
        world_axes.setLightOff()
        world_axes.reparentTo(self.render)
        for txt, pos, col in (("E", (3.3, 0, 0), (1, .4, .4, 1)), ("N", (0, 3.3, 0), (.4, 1, .4, 1)),
                              ("Up", (0, 0, 3.3), (.5, .7, 1, 1))):
            tn = TextNode("lbl")
            tn.setText(txt)
            tn.setTextColor(*col)
            tn.setAlign(TextNode.ACenter)
            t = self.render.attachNewNode(tn)
            t.setPos(*pos)
            t.setScale(0.5)
            t.setBillboardPointEye()
            t.setLightOff()

        self.phone_pos_np = self.render.attachNewNode("phone_pos")
        self.phone = build_phone(args.scale)
        self.phone.reparentTo(self.phone_pos_np)
        self.shadow = make_lines([((-0.3, 0, 0), (0.3, 0, 0), (1, 1, 0, 0.6)),
                                  ((0, -0.3, 0), (0, 0.3, 0), (1, 1, 0, 0.6))])
        self.shadow.setLightOff()
        self.shadow.reparentTo(self.render)
        self.drop_line = None
        self.accel_np = None
        self.trail_np = None
        self.fix_np = None

        # state
        self.mode = args.mode         # "ekf" or "gps"
        self.target = Vec3(0, 0, 0)   # GPS-only mode: last fix (m, ENU)
        self.pos = Vec3(0, 0, 0)      # displayed position
        self.corr = Vec3(0, 0, 0)     # EKF mode: GPS correction being blended in smoothly
        self.fixes = deque(maxlen=3000)
        self.fixes_dirty = False
        self.show_fixes = True
        self.fstate = None
        self.trail = deque(maxlen=5000)
        self.trail_dirty = False
        self.show_trail = True
        self.show_accel = True
        self.follow = True
        self.gps_info = None

        # camera orbit
        self.cam_h, self.cam_p, self.cam_dist = 30.0, -25.0, 6.0
        self.cam_center = Vec3(0, 0, 0)
        self.dragging = False
        self.last_mouse = None
        self.accept("mouse1", self._drag_start)
        self.accept("mouse1-up", self._drag_end)
        self.accept("wheel_up", self._zoom, [0.88])
        self.accept("wheel_down", self._zoom, [1.0 / 0.88])
        self.accept("f", self._toggle, ["follow"])
        self.accept("t", self._toggle, ["show_trail"])
        self.accept("a", self._toggle, ["show_accel"])
        self.accept("g", self._toggle, ["show_fixes"])
        self.accept("k", self._toggle_mode)
        self.accept("r", self.reset_origin)
        self.accept("escape", self.userExit)

        self.hud = OnscreenText(text="", pos=(-1.3, 0.92), scale=0.045, fg=(0.9, 0.95, 1, 1),
                                align=TextNode.ALeft, mayChange=True, font=self.loader.loadFont("cmtt12"))
        self.hud.reparentTo(self.a2dTopLeft)
        self.hud.setPos(0.04, -0.07)
        help_txt = OnscreenText(text="drag: orbit  wheel: zoom  F: follow  R: reset  T: trail  A: accel  K: EKF/GPS  G: GPS fixes  Esc: quit",
                                scale=0.04, fg=(0.6, 0.6, 0.7, 1), align=TextNode.ALeft)
        help_txt.reparentTo(self.a2dBottomLeft)
        help_txt.setPos(0.04, 0.04)

        self.taskMgr.add(self.update, "update")

    # ---------------------------------------------------------------- input
    def _toggle(self, name):
        setattr(self, name, not getattr(self, name))
        self.trail_dirty = True
        self.fixes_dirty = True

    def _toggle_mode(self):
        self.mode = "gps" if self.mode == "ekf" else "ekf"
        self.corr = Vec3(0, 0, 0)

    def _zoom(self, f):
        self.cam_dist = min(max(self.cam_dist * f, 0.3), 2000)

    def _drag_start(self):
        if self.mouseWatcherNode.hasMouse():
            self.dragging = True
            m = self.mouseWatcherNode.getMouse()
            self.last_mouse = (m.x, m.y)

    def _drag_end(self):
        self.dragging = False

    def reset_origin(self):
        self.rx.reset_fusion()
        self.target = Vec3(0, 0, 0)
        self.pos = Vec3(0, 0, 0)
        self.corr = Vec3(0, 0, 0)
        self.trail.clear()
        self.fixes.clear()
        self.trail_dirty = True
        self.fixes_dirty = True

    # ---------------------------------------------------------------- frame
    def update(self, task):
        dt = min(self.clock.getDt(), 0.1)
        pkt, sender, last_rx, rate, new_gps = self.rx.snapshot()

        # camera orbit drag
        if self.dragging and self.mouseWatcherNode.hasMouse():
            m = self.mouseWatcherNode.getMouse()
            dx, dy = m.x - self.last_mouse[0], m.y - self.last_mouse[1]
            self.last_mouse = (m.x, m.y)
            self.cam_h -= dx * 120
            self.cam_p = min(max(self.cam_p - dy * 80, -89), 89)

        height_src = "-"
        q = None
        fs = self.fstate = self.rx.fusion_state()
        if pkt is not None:
            for gps in new_gps:
                self.gps_info = gps
            height_src = "baro" if ("p" in pkt and self.args.alt != "gps") else ("gps" if self.gps_info else "-")
            if "q" in pkt:
                w, x, y, z = pkt["q"]
                q = Quat(w, x, y, z)
                q.normalize()
                self.phone.setQuat(q)
        for fx in fs["new_fixes"]:
            self.fixes.append(Vec3(*fx))
            self.target = Vec3(fx[0], fx[1], 0)
            self.fixes_dirty = True

        fused = Vec3(*fs["pos"])
        if self.mode == "ekf":
            # Show the filter output; GPS corrections are blended in over ~0.5 s instead of jumping.
            self.corr -= Vec3(*fs["jump"])
            self.corr *= math.exp(-dt / 0.5)
            self.pos = fused + self.corr
        else:
            # GPS-only: smooth toward the last fix (height still from the filter / barometer)
            self.target.z = fused.z
            k = 1.0 - math.exp(-dt * self.args.smooth)
            self.pos += (self.target - self.pos) * k
        self.phone_pos_np.setPos(self.pos)
        self.shadow.setPos(self.pos.x, self.pos.y, 0.001)

        if self.drop_line is not None:
            self.drop_line.removeNode()
        self.drop_line = make_lines([((self.pos.x, self.pos.y, 0), (self.pos.x, self.pos.y, self.pos.z),
                                      (1, 1, 0, 0.5))], 1)
        self.drop_line.setLightOff()
        self.drop_line.reparentTo(self.render)

        # trail
        if pkt is not None and (not self.trail or (self.trail[-1] - self.pos).length() > 0.05):
            self.trail.append(Vec3(self.pos))
            self.trail_dirty = True
        if self.trail_dirty:
            if self.trail_np is not None:
                self.trail_np.removeNode()
                self.trail_np = None
            if self.show_trail and len(self.trail) > 1:
                ls = LineSegs()
                ls.setThickness(2)
                ls.setColor(1, 0.6, 0.1, 1)
                ls.moveTo(self.trail[0])
                for p in list(self.trail)[1:]:
                    ls.drawTo(p)
                self.trail_np = NodePath(ls.create())
                self.trail_np.setLightOff()
                self.trail_np.reparentTo(self.render)
            self.trail_dirty = False

        # raw GPS fixes
        if self.fixes_dirty:
            if self.fix_np is not None:
                self.fix_np.removeNode()
                self.fix_np = None
            if self.show_fixes and self.fixes:
                ls = LineSegs()
                ls.setThickness(2)
                ls.setColor(0.2, 0.9, 1, 0.9)
                r = 0.15
                for p in self.fixes:
                    ls.moveTo(p.x - r, p.y, p.z)
                    ls.drawTo(p.x + r, p.y, p.z)
                    ls.moveTo(p.x, p.y - r, p.z)
                    ls.drawTo(p.x, p.y + r, p.z)
                self.fix_np = NodePath(ls.create())
                self.fix_np.setLightOff()
                self.fix_np.reparentTo(self.render)
            self.fixes_dirty = False

        # linear acceleration vector in world frame
        if self.accel_np is not None:
            self.accel_np.removeNode()
            self.accel_np = None
        lin_w = None
        if pkt is not None and q is not None and "lin" in pkt:
            lin_w = q.xform(Vec3(*pkt["lin"]))
            if self.show_accel:
                self.accel_np = make_lines([((0, 0, 0), tuple(lin_w * self.args.accel_scale), (1, 0.2, 1, 1))], 4)
                self.accel_np.setLightOff()
                self.accel_np.setDepthTest(False)
                self.accel_np.setBin("fixed", 20)
                self.accel_np.reparentTo(self.phone_pos_np)

        # camera
        center = self.pos if self.follow else Vec3(0, 0, 0)
        self.cam_center += (center - self.cam_center) * (1.0 - math.exp(-dt * 6))
        hr, pr = math.radians(self.cam_h), math.radians(self.cam_p)
        off = Vec3(-math.sin(hr) * math.cos(pr), -math.cos(hr) * math.cos(pr), -math.sin(pr)) * self.cam_dist
        self.camera.setPos(self.cam_center + off)
        self.camera.lookAt(self.cam_center)

        self.hud.setText(self._hud_text(pkt, sender, last_rx, rate, height_src, lin_w))
        return task.cont

    def _hud_text(self, pkt, sender, last_rx, rate, height_src, lin_w):
        ph = self.rx.phone
        hello = ""
        if ph:
            hello = f"  hello->{ph[0]}:{ph[1]}"
        elif self.rx.phone_auto:
            hello = "  hello: phone IP not found via ADB (adb devices?)"
        if self.rx.phone_mode == "auto" and self.rx.discovery_status and pkt is None:
            hello = "  " + self.rx.discovery_status
        lines = [f"UDP :{self.args.port}" + hello]
        if pkt is None:
            lines.append("waiting for packets...")
            return "\n".join(lines)
        age = time.time() - last_rx
        status = "LIVE" if age < 1.0 else f"STALE {age:.0f}s"
        lines[0] += f"  from {sender[0]}  [{status}]  {rate:5.1f} pkt/s  lost {self.rx.lost}"
        hpr = self.phone.getHpr(self.render)
        lines.append(f"heading {(-hpr[0]) % 360:6.1f} deg   pitch {hpr[1]:+6.1f}   roll {hpr[2]:+6.1f}")
        if "q" in pkt:
            lines.append("quat  w{:+.3f} x{:+.3f} y{:+.3f} z{:+.3f}".format(*pkt["q"]))
        if "acc" in pkt:
            a = pkt["acc"]
            lines.append(f"acc  {a[0]:+7.2f} {a[1]:+7.2f} {a[2]:+7.2f} m/s2  |{math.sqrt(sum(v*v for v in a)):.2f}|")
        if "gyr" in pkt:
            lines.append("gyr  {:+7.2f} {:+7.2f} {:+7.2f} rad/s".format(*pkt["gyr"]))
        if "mag" in pkt:
            lines.append("mag  {:+7.1f} {:+7.1f} {:+7.1f} uT".format(*pkt["mag"]))
        if lin_w is not None:
            lines.append(f"lin(world) E{lin_w.x:+6.2f} N{lin_w.y:+6.2f} U{lin_w.z:+6.2f} m/s2")
        if "p" in pkt:
            lines.append(f"pressure {pkt['p']:.2f} hPa")
        g = self.gps_info
        if g:
            spd = g.get("spd")
            lines.append(f"GPS {g['lat']:.7f}, {g['lon']:.7f}  alt {g.get('alt') or 0:.1f} m  "
                         f"acc {g.get('acc') or 0:.1f} m  spd {(spd or 0) * 3.6:.1f} km/h [{g.get('prov', '')}]")
        else:
            lines.append("GPS: no fix")
        lines.append(f"pos  E{self.pos.x:+8.2f}  N{self.pos.y:+8.2f}  U{self.pos.z:+7.2f} m  (height: {height_src})")
        fs = self.fstate
        if fs is not None:
            v = fs["vel"]
            age = fs["gps_age"]
            age_txt = "no fix" if age is None else f"{age:4.1f}s ago (every {fs['gps_interval']:.1f}s)"
            lines.append(f"vel  E{v[0]:+6.2f} N{v[1]:+6.2f} U{v[2]:+6.2f} m/s  |{math.hypot(v[0], v[1]) * 3.6:5.1f} km/h|"
                         f"  {'STILL' if fs['still'] else ('WALKING' if fs['walking'] else 'standing')}")
            cam = "off" if fs["ar_state"] is None else fs["ar_state"]
            src = "CAMERA" if fs["ar_tracking"] else "PDR"
            lines.append(f"motion src {src}  camera {cam}  steps {fs['steps']}  step {fs['step_len']:.2f} m")
            lines.append(f"mode {self.mode.upper()}  GPS {age_txt}  pos sigma {fs['pos_sigma']:.1f} m"
                         f"  bias {fs['bias'][0]:+.2f} {fs['bias'][1]:+.2f} {fs['bias'][2]:+.2f}")
        lines.append(f"follow {'on' if self.follow else 'off'}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- Windows firewall

def _fw_rule_name(port):
    return f"SmartLogger-UDP-{port}"


def firewall_rule_exists(port):
    r = subprocess.run(["netsh", "advfirewall", "firewall", "show", "rule", f"name={_fw_rule_name(port)}"],
                       capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return r.returncode == 0


def ensure_firewall(port):
    """On Windows, offer to add an inbound allow rule for the UDP port (asks via dialog + UAC)."""
    if sys.platform != "win32":
        return
    try:
        if firewall_rule_exists(port):
            print(f"[firewall] rule '{_fw_rule_name(port)}' already exists")
            return
    except OSError as e:
        print(f"[firewall] cannot query rules: {e}")
        return
    import ctypes
    MB_YESNO, MB_ICONQUESTION, MB_ICONINFO, MB_ICONWARN, MB_TOPMOST, IDYES = 0x4, 0x20, 0x40, 0x30, 0x40000, 6
    box = ctypes.windll.user32.MessageBoxW
    msg = (f"スマホからのデータを受信するため、Windows ファイアウォールで\n"
           f"UDP ポート {port} の受信を許可しますか？\n\n"
           f"「はい」を押すと管理者権限の確認 (UAC) が表示されます。\n"
           f"（ルール名: {_fw_rule_name(port)}、全ネットワークプロファイル）")
    if box(None, msg, "SmartLogger - ファイアウォール", MB_YESNO | MB_ICONQUESTION | MB_TOPMOST) != IDYES:
        print("[firewall] skipped by user")
        return
    netsh_args = (f'advfirewall firewall add rule name={_fw_rule_name(port)} dir=in action=allow '
                  f'protocol=UDP localport={port} profile=any')
    ps = (f"Start-Process -FilePath netsh -ArgumentList '{netsh_args}' "
          f"-Verb RunAs -Wait -WindowStyle Hidden")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if firewall_rule_exists(port):
        print(f"[firewall] added rule '{_fw_rule_name(port)}'")
        box(None, f"UDP ポート {port} の受信を許可しました。", "SmartLogger - ファイアウォール",
            MB_ICONINFO | MB_TOPMOST)
    else:
        print("[firewall] rule was not added (UAC cancelled?)")
        box(None, "ファイアウォールのルールを追加できませんでした。\n"
                  "（UAC がキャンセルされた可能性があります）", "SmartLogger - ファイアウォール",
            MB_ICONWARN | MB_TOPMOST)


def _adb(*args, timeout=10):
    return subprocess.run(["adb", *args], capture_output=True, text=True, timeout=timeout,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout


def phone_ip_from_adb():
    """Return the Wi-Fi IPv4 address of an ADB-connected phone, or None.

    Works when several ADB entries exist (wireless debugging often lists the same phone twice,
    which makes a plain `adb shell` fail with "more than one device").
    """
    import re
    try:
        serials = [ln.split()[0] for ln in _adb("devices").splitlines()[1:]
                   if ln.strip() and ln.split()[-1] == "device"]
        for serial in serials:
            m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", _adb("-s", serial, "shell", "ip", "-4", "addr", "show", "wlan0"))
            if m:
                return m.group(1)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description="SmartLogger real-time 3D viewer")
    ap.add_argument("--port", type=int, default=5005, help="UDP port to listen on (default 5005)")
    ap.add_argument("--scale", type=float, default=8.0, help="phone model scale factor vs. real size (default 8)")
    ap.add_argument("--alt", choices=["baro", "gps"], default="baro", help="height source (default baro)")
    ap.add_argument("--smooth", type=float, default=4.0, help="GPS-only mode smoothing rate 1/s (default 4)")
    ap.add_argument("--mode", choices=["ekf", "gps"], default="ekf",
                    help="position: ekf = accel dead reckoning + GPS fusion (default), gps = GPS only")
    ap.add_argument("--accel-scale", type=float, default=0.3, help="metres per m/s2 for accel arrow")
    ap.add_argument("--grid", type=float, default=100, help="half size of ground grid in metres")
    ap.add_argument("--no-firewall", action="store_true", help="skip the Windows firewall check")
    ap.add_argument("--phone-port", type=int, default=None, help="the app's UDP port if different from --port")
    ap.add_argument("--record", default=None, help="append every received packet to this JSONL file")
    ap.add_argument("--phone", default="auto",
                    help="how to reach the phone (needed behind ESET-like firewalls): 'auto' = search the "
                         "LAN (broadcast, ARP table, subnet sweep; default), 'adb' = IP via ADB, "
                         "'none' = only listen, or a fixed IP[:port]")
    args = ap.parse_args()
    if not args.no_firewall:
        ensure_firewall(args.port)
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind(("0.0.0.0", args.port))
    except OSError:
        sys.exit(f"UDP port {args.port} is already in use - is another viewer.py still running? "
                 f"Close it or use --port.")
    finally:
        probe.close()
    Viewer(args).run()


if __name__ == "__main__":
    main()
