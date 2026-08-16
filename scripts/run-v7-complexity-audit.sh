#!/usr/bin/env bash
set -euo pipefail
REPOSITORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT="8013300c23553928a0269e6be27f5baaedee7e53"
cd "$REPOSITORY"
git cat-file -e "${PARENT}^{commit}"
git diff --quiet "$PARENT" -- results/generalized_entropic_consensus_v6
test -s results/complexity_entropic_coordination_v7/development/audits/v6_complexity_audit.csv
test -s notes/68_v6_environment_complexity_audit.md
echo "V6 parent and frozen namespace verified; V7 complexity audit is present."
