"""Find the phone on the local network without ADB.

The viewer sends a small UDP "hello" to candidate addresses; the SmartLogger app answers any
hello by streaming to the sender. Because the PC starts the exchange, the replies also pass
stateful PC firewalls (ESET etc.) that drop unsolicited inbound UDP.

Candidates, cheapest first:
  1. directed broadcast of every local IPv4 subnet (e.g. 192.168.128.255)
  2. neighbours in the ARP table that are inside those subnets
  3. every host address of each subnet (only for subnets of /22 or smaller)

Virtual adapters (VirtualBox, Hyper-V, WSL, VPN, Tailscale ...) are skipped.
"""

import ipaddress
import re
import socket
import subprocess
import sys

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_SKIP_ADAPTER = re.compile(r"virtualbox|vmware|hyper-v|vethernet|wsl|tailscale|wireguard|vpn|"
                           r"loopback|bluetooth|tap-|tunnel", re.I)
MAX_SWEEP_HOSTS = 1024


def _run(cmd, timeout=10):
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=_NO_WINDOW).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.decode("utf-8", "ignore") if b"\x00" not in out else out.decode("utf-16-le", "ignore")


def _usable(iface):
    ip = iface.ip
    return not (ip.is_loopback or ip.is_link_local or ip.is_multicast
                or ip in ipaddress.ip_network("100.64.0.0/10")      # CGNAT / Tailscale
                or iface.network.prefixlen < 16)


def local_subnets():
    """List of ipaddress.IPv4Interface for the PC's real LAN interfaces."""
    ifaces = []
    if sys.platform == "win32":
        ps = ("Get-NetIPAddress -AddressFamily IPv4 | ForEach-Object { "
              "$a = Get-NetAdapter -InterfaceIndex $_.InterfaceIndex -ErrorAction SilentlyContinue; "
              "if ($a -and $a.Status -eq 'Up') { \"$($_.IPAddress)/$($_.PrefixLength)|$($a.InterfaceDescription)\" } }")
        for line in _run(["powershell", "-NoProfile", "-Command", ps]).splitlines():
            addr, _, desc = line.strip().partition("|")
            if not addr or _SKIP_ADAPTER.search(desc):
                continue
            try:
                ifaces.append(ipaddress.IPv4Interface(addr))
            except ValueError:
                pass
    if not ifaces:
        # fallback: the interface used for the default route, assumed /24
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("192.0.2.1", 9))   # TEST-NET, nothing is sent
            ifaces.append(ipaddress.IPv4Interface(s.getsockname()[0] + "/24"))
            s.close()
        except OSError:
            pass
    return [i for i in ifaces if _usable(i)]


def arp_neighbors():
    """IPv4 addresses currently in the ARP cache (unicast only)."""
    ips = set()
    for m in re.finditer(r"(\d{1,3}(?:\.\d{1,3}){3})\s+([0-9a-fA-F]{2}(?:[-:][0-9a-fA-F]{2}){5})", _run(["arp", "-a"])):
        try:
            ip = ipaddress.IPv4Address(m.group(1))
        except ValueError:
            continue
        mac = m.group(2).lower().replace(":", "-")
        if ip.is_multicast or mac == "ff-ff-ff-ff-ff-ff":
            continue
        ips.add(str(ip))
    return ips


class Discovery:
    """Produces the list of addresses to say hello to in each discovery round."""

    def __init__(self):
        self.round = 0
        self.subnets = []
        self.status = "not started"

    def targets(self):
        """Addresses for this round (call about every 2 s while no phone is known)."""
        if self.round % 15 == 0:      # refresh interfaces every ~30 s (Wi-Fi may change)
            self.subnets = local_subnets()
        own = {str(i.ip) for i in self.subnets}
        out = []
        for i in self.subnets:
            out.append(str(i.network.broadcast_address))
        neigh = [ip for ip in sorted(arp_neighbors())
                 if ip not in own and any(ipaddress.IPv4Address(ip) in i.network for i in self.subnets)
                 and not any(ip == str(i.network.broadcast_address) for i in self.subnets)]
        out += neigh
        sweep = 0
        if self.round % 3 == 2:       # full sweep every 3rd round
            for i in self.subnets:
                if i.network.num_addresses - 2 <= MAX_SWEEP_HOSTS:
                    for h in i.network.hosts():
                        h = str(h)
                        if h not in own:
                            out.append(h)
                            sweep += 1
        nets = ", ".join(str(i.network) for i in self.subnets) or "none"
        self.status = (f"searching {nets}: broadcast + {len(neigh)} ARP"
                       + (f" + sweep {sweep}" if sweep else ""))
        self.round += 1
        return list(dict.fromkeys(out))


if __name__ == "__main__":
    d = Discovery()
    for _ in range(3):
        t = d.targets()
        print(d.status, "->", len(t), "targets, first:", t[:6])
