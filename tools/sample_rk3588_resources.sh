#!/usr/bin/env bash
# Sample one RK3588 sender process without fabricating accelerator utilization.
# Optional NPU_UTIL_CMD and VPU_UTIL_CMD must print one numeric percent value.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <sender-pid> <output.csv> [interval-seconds]" >&2
  exit 2
fi

pid="$1"
output="$2"
interval="${3:-1}"
if ! [[ "$pid" =~ ^[1-9][0-9]*$ ]] || ! [[ "$interval" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "sender-pid must be positive and interval-seconds must be numeric" >&2
  exit 2
fi

mkdir -p "$(dirname "$output")"
printf 'monotonic_ms,cpu_percent,npu_percent,vpu_percent\n' >"$output"

process_cpu_ticks() {
  # Strip the potentially space-containing comm field before addressing the
  # utime/stime fields (12 and 13 after the closing parenthesis).
  awk '{ tail = substr($0, index($0, ")") + 2); split(tail, fields, " "); print fields[12] + fields[13] }' \
    "/proc/$pid/stat" 2>/dev/null
}

uptime_seconds() {
  awk '{print $1}' /proc/uptime 2>/dev/null
}

percent_from_command() {
  local command="$1"
  local value
  if [[ -z "$command" ]]; then
    return 0
  fi
  value="$(sh -c "$command" 2>/dev/null | tr -d '[:space:]')" || return 0
  if [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    printf '%s' "$value"
  fi
}

last_ticks="$(process_cpu_ticks)"
last_uptime="$(uptime_seconds)"
clock_ticks="$(getconf CLK_TCK 2>/dev/null || printf '100')"
while kill -0 "$pid" 2>/dev/null; do
  sleep "$interval"
  current_ticks="$(process_cpu_ticks)"
  current_uptime="$(uptime_seconds)"
  if [[ -z "$current_ticks" || -z "$current_uptime" ]]; then
    break
  fi
  timestamp_ms="$(awk -v value="$current_uptime" 'BEGIN { printf "%.0f", value * 1000 }')"
  cpu_percent="$(awk -v ticks0="$last_ticks" -v ticks1="$current_ticks" -v time0="$last_uptime" \
    -v time1="$current_uptime" -v hz="$clock_ticks" \
    'BEGIN { elapsed = time1 - time0; if (elapsed > 0) printf "%.3f", (ticks1 - ticks0) * 100 / hz / elapsed; else print "" }')"
  npu_percent="$(percent_from_command "${NPU_UTIL_CMD:-}")"
  vpu_percent="$(percent_from_command "${VPU_UTIL_CMD:-}")"
  printf '%s,%s,%s,%s\n' "$timestamp_ms" "$cpu_percent" "$npu_percent" "$vpu_percent" >>"$output"
  last_ticks="$current_ticks"
  last_uptime="$current_uptime"
done
