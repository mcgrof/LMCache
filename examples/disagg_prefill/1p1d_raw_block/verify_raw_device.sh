#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

device=${1:?"usage: verify_raw_device.sh /dev/disk/by-id/<dedicated-device>"}

if [[ ! $device =~ ^/dev/disk/by-id/[A-Za-z0-9._:+-]+$ ]]; then
    echo "Refusing non-persistent device path: $device" >&2
    exit 1
fi

resolved=$(readlink -f -- "$device" 2>/dev/null || true)
block_name=$(basename -- "${resolved:-$device}")

# Every other check in this script asks whether the device is in use. None of
# them can ask whether it matters: a pristine reference drive kept for
# comparison is blank, unmounted, unheld and signature-free, so it passes all
# of them. An operator lists such devices here and they are refused by name,
# before anything else, so the refusal says why.
protected_file=${LMCACHE_PROTECTED_DEVICES_FILE:-/etc/lmcache/protected-devices}
if [[ -n ${LMCACHE_PROTECTED_DEVICES_FILE:-} && ! -r $protected_file ]]; then
    echo "Cannot read the protected-device list: $protected_file" >&2
    exit 1
fi
if [[ -r $protected_file ]]; then
    serial_path=/sys/class/block/$block_name/device/serial
    wwid_path=/sys/class/block/$block_name/wwid
    device_serial=""
    device_wwid=""
    [[ -r $serial_path ]] && device_serial=$(tr -d '[:space:]' < "$serial_path")
    [[ -r $wwid_path ]] && device_wwid=$(tr -d '[:space:]' < "$wwid_path")
    while IFS= read -r entry; do
        entry=${entry%%#*}
        entry=$(printf '%s' "$entry" | tr -d '[:space:]')
        [[ -z $entry ]] && continue
        entry_resolved=$(readlink -f -- "$entry" 2>/dev/null || true)
        if [[ $entry == "$device" ]] \
            || [[ -n $entry_resolved && $entry_resolved == "$resolved" ]] \
            || [[ -n $device_serial && $entry == "$device_serial" ]] \
            || [[ -n $device_wwid && $entry == "$device_wwid" ]]; then
            echo "Refusing a device listed as protected in $protected_file" >&2
            echo "  requested: $device -> ${resolved:-<unresolved>}" >&2
            exit 1
        fi
    done < "$protected_file"
fi

if [[ ! -b "$resolved" ]]; then
    echo "Refusing non-block device: $device -> ${resolved:-<unresolved>}" >&2
    exit 1
fi

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
echo "This means the device is unused and confirmed, not that its"
echo "contents are expendable. Keep $protected_file current."
