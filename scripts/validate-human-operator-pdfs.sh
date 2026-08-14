#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/human-operator-common.sh"
run_human_command analysis/pdf_validation.log validate-pdfs
