#!/bin/sh
set -e

# Corrige permissões dos diretórios montados via bind mount antes de executar como appuser.
# Necessário porque o host cria ./artifacts e ./data como root, mas o container roda como appuser (UID 1001).
chown -R appuser:appgroup /app/artifacts /app/data 2>/dev/null || true

exec gosu appuser "$@"
