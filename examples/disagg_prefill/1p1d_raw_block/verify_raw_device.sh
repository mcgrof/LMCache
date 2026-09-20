#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

device=${1:?"usage: verify_raw_device.sh /dev/disk/by-id/<dedicated-device>"}

if [[ ! $device =~ ^/dev/disk/by-id/[A-Za-z0-9._:+-]+$ ]]; then
    echo "Refusing non-persistent device path: $device" >&2
    exit 1
fi

resolved=$(readlink -f -- "$device")
if [[ ! -b "$resolved" ]]; then
    echo "Refusing non-block device: $device -> $resolved" >&2
    exit 1
fi

block_name=$(basename -- "$resolved")
if [[ -e "/sys/class/block/$block_name/partition" ]]; then
    echo "Refusing partition device: $resolved" >&2
    exit 1
fi

block_type=$(lsblk -dnro TYPE "$resolved")
if [[ $block_type != disk ]]; then
    echo "Refusing non-disk block stack type '$block_type': $resolved" >&2
    exit 1
fi

mapfile -t block_tree < <(lsblk -nrpo NAME "$resolved")
if (( ${#block_tree[@]} != 1 )); then
    echo "Refusing device with partitions or child block devices: $resolved" >&2
    printf '  %s\n' "${block_tree[@]}" >&2
    exit 1
fi

if lsblk -nrpo MOUNTPOINT "$resolved" | grep -q '[^[:space:]]'; then
    echo "Refusing mounted device: $resolved" >&2
    exit 1
fi

if compgen -G "/sys/class/block/$block_name/holders/*" >/dev/null; then
    echo "Refusing device held by another block stack: $resolved" >&2
    exit 1
fi

set +e
blkid -p "$resolved" >/dev/null 2>&1
blkid_status=$?
set -e
case $blkid_status in
    0)
        echo "Refusing device with a recognized filesystem or RAID signature" >&2
        exit 1
        ;;
    2) ;;
    *)
        echo "Could not inspect signatures on $resolved (blkid=$blkid_status)" >&2
        exit 1
        ;;
esac

if [[ ${LMCACHE_CONFIRM_RAW_DEVICE_ERASE:-} != "$device" ]]; then
    echo "Raw-block P/D overwrites metadata and payload ranges on $device." >&2
    echo "Set LMCACHE_CONFIRM_RAW_DEVICE_ERASE=$device to confirm this exact path." >&2
    exit 1
fi

echo "Raw-device preflight passed: $device -> $resolved"
