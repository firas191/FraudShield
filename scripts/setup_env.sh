#!/usr/bin/env bash
# FraudShield one-shot environment setup + verification (Week 1).
# Run inside WSL Ubuntu from the repo root:
#   bash scripts/setup_env.sh
# Safe to re-run: every step is idempotent and skips work already done.

set -uo pipefail

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# --- 1. Network ---------------------------------------------------------
echo "== 1/6 Network check =="
if ! ping -c 1 -W 3 1.1.1.1 >/dev/null 2>&1; then
    fail "No internet routing. Run 'wsl.exe --shutdown' (then reopen Ubuntu) and check Windows Wi-Fi/VPN, then re-run this script."
fi
ok "routing works"

if ! curl -sI --max-time 10 https://pypi.org >/dev/null 2>&1; then
    warn "DNS broken — pointing /etc/resolv.conf at 1.1.1.1 (needs sudo)"
    echo "nameserver 1.1.1.1" | sudo tee /etc/resolv.conf >/dev/null
    curl -sI --max-time 10 https://pypi.org >/dev/null 2>&1 || fail "still no DNS after fix — restart WSL ('wsl.exe --shutdown') and re-run"
fi
ok "DNS + TLS to pypi.org works"

# --- 2. Python 3.11 -----------------------------------------------------
echo "== 2/6 Python 3.11 =="
if ! command -v python3.11 >/dev/null 2>&1; then
    warn "python3.11 missing — installing via deadsnakes (needs sudo)"
    sudo apt-get install -y software-properties-common >/dev/null
    sudo add-apt-repository -y ppa:deadsnakes/ppa >/dev/null
    sudo apt-get update -qq
    sudo apt-get install -y python3.11 python3.11-venv git >/dev/null
fi
ok "$(python3.11 --version)"

# --- 3. Virtualenv ------------------------------------------------------
echo "== 3/6 Virtualenv =="
cd "$(dirname "$0")/.."   # repo root, wherever the script is invoked from
[ -d .venv ] || python3.11 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
ok "venv active: $(python --version) at $VIRTUAL_ENV"

# --- 4. Dependencies (resumable-ish: retries + long timeout) ------------
echo "== 4/6 Dependencies (first run downloads ~3.5GB — be patient) =="
pip install --quiet --upgrade pip
pip install --timeout 180 --retries 10 -r requirements.txt || \
    fail "pip install failed — usually a dropped connection; just re-run this script, pip's cache keeps finished wheels"
ok "all dependencies installed"

# --- 5. CUDA ------------------------------------------------------------
echo "== 5/6 CUDA check =="
CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_OK" = "True" ]; then
    ok "torch $(python -c 'import torch; print(torch.__version__)') sees GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
else
    warn "torch installed but CUDA unavailable — training will use CPU (fine for Week 1; check 'nvidia-smi' works and re-install torch if you want GPU)"
fi

# --- 6. Test suite ------------------------------------------------------
echo "== 6/6 Week 1 test suite =="
pytest tests/ -v || fail "tests failed — paste the output into the chat"
ok "Week 1 verification complete. Next: python -m data.prepare_dataset download && python -m data.prepare_dataset explore"
