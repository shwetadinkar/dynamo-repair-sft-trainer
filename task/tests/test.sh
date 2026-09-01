#!/bin/bash
set -u
umask 077

mkdir -p /logs/verifier
rm -f /logs/verifier/reward.txt /logs/verifier/reward.tmp

# Pytest needs verifier files; package code runs as uid 65532 and must not.
chown -R 0:0 /tests /logs/verifier
chmod -R go-rwx /tests /logs/verifier

cd /
env -i \
    HOME=/root \
    LANG=C.UTF-8 \
    PATH=/usr/local/bin:/usr/bin:/bin \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    pytest --ctrf /logs/verifier/ctrf.json \
           --rootdir=/tests --confcutdir=/tests --noconftest -p no:cacheprovider \
           /tests/test_outputs.py -rA
status=$?

if [ "$status" -eq 0 ]; then
  printf '1\n' > /logs/verifier/reward.tmp
else
  printf '0\n' > /logs/verifier/reward.tmp
fi
mv /logs/verifier/reward.tmp /logs/verifier/reward.txt
exit "$status"
