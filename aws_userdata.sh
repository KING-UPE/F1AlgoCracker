#!/bin/bash
# ============================================================
#  AWS EC2 User-Data Bootstrap Script
#  F1 Constants Cracker — Box Box Box Challenge
#
#  Recommended instance:  c6i.32xlarge  (128 vCPUs, 256 GB RAM)
#  AMI:                   Amazon Linux 2023 (al2023-ami-*)
#  Spot pricing:          ~$0.60–$1.20 / hr  (vs $5.44 on-demand)
#  Expected runtime:      3–8 minutes with 128 cores
#
#  HOW TO USE:
#    1. Launch EC2 (see aws_launch_commands.sh for CLI commands)
#    2. Paste this script into "User data" when launching
#    3. SSH in and tail the log:
#         ssh ec2-user@<IP> "tail -f /home/ec2-user/crack.log"
#    4. When complete, copy constants.json back:
#         scp ec2-user@<IP>:~/box-box-box/solution/constants.json solution/
# ============================================================

set -euo pipefail
LOG="/home/ec2-user/crack.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================"
echo "  Box Box Box — EC2 Bootstrap Starting"
echo "  $(date)"
echo "  Instance: $(curl -s http://169.254.169.254/latest/meta-data/instance-type)"
echo "  vCPUs:    $(nproc)"
echo "============================================"

# ── System packages ──────────────────────────
dnf update -y -q
dnf install -y git python3-pip python3-devel gcc-c++ -q

# ── Python dependencies ──────────────────────
pip3 install -q --upgrade pip
pip3 install -q numpy scipy

echo "Python & scipy installed: $(python3 -c 'import scipy; print(scipy.__version__)')"

# ── Clone your repo ──────────────────────────
# EDIT THIS: replace with your actual GitHub fork URL
REPO_URL="${REPO_URL:-https://github.com/YOUR_USERNAME/box-box-box.git}"
REPO_DIR="/home/ec2-user/box-box-box"

cd /home/ec2-user
git clone "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"

echo "Repo cloned: $(pwd)"
echo "Data files:  $(ls data/historical_races/ | wc -l) files"

# ── Copy cracker script ───────────────────────
# (or it's already in repo at solution/crack_constants.py)
# cp /path/to/crack_constants.py "$REPO_DIR/"

# ── Run the cracker ───────────────────────────
NCPUS=$(nproc)
echo ""
echo "Starting constants cracker with $NCPUS CPUs and 2000 training races..."
echo "$(date)"

python3 crack_constants.py 2000

echo ""
echo "============================================"
echo "  COMPLETED at $(date)"
echo "  Constants saved to solution/constants.json"
echo "============================================"

cat solution/constants.json

# ── Optional: Validate on test cases ─────────
echo ""
echo "Running test validation..."
PASS=0; FAIL=0
for f in data/test_cases/inputs/test_*.json; do
    id=$(basename "$f" .json)
    expected="data/test_cases/expected_outputs/${id}.json"
    if [ -f "$expected" ]; then
        pred=$(python3 solution/race_simulator.py < "$f")
        exp_order=$(python3 -c "import json,sys; d=json.load(open('$expected')); print(' '.join(d['finishing_positions']))")
        pred_order=$(echo "$pred" | python3 -c "import json,sys; d=json.load(sys.stdin); print(' '.join(d['finishing_positions']))")
        if [ "$pred_order" = "$exp_order" ]; then
            PASS=$((PASS+1))
        else
            FAIL=$((FAIL+1))
        fi
    fi
done

TOTAL=$((PASS+FAIL))
if [ $TOTAL -gt 0 ]; then
    echo "Test results: $PASS/$TOTAL passed ($(python3 -c "print(f'{$PASS/$TOTAL*100:.1f}')") %)"
fi

echo ""
echo "Done! To copy constants back:"
echo "  scp ec2-user@\$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):~/box-box-box/solution/constants.json solution/"
