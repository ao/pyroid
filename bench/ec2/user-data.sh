#!/bin/bash
shutdown -h +420 "pyroid-bench: 7-hour lifetime reached"
# pyroid-bench (bench/ec2/README.md). The line above runs first: whatever happens, the instance powers off after
# 7 hours, and its instance-initiated shutdown behaviour (terminate) turns that into termination, deleting its root
# volume (DeleteOnTermination). No SSH, no key pair, no inbound rules: access is SSM send-command only.
mkdir -p /opt/pyroid/logs
date -u +%Y-%m-%dT%H:%M:%SZ >/opt/pyroid/logs/launched-at
cat /run/systemd/shutdown/scheduled >/opt/pyroid/logs/shutdown-scheduled 2>&1 || true
