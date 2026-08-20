#!/bin/bash
# Wire the local vLLM server (127.0.0.1:8000) behind this the target hardware instance's
# authed a reverse proxy edge, on the first free "normal" external port. Run on the
# instance itself, after scripts/serve.sh is already running.
#
# the target hardware external ports are fixed at instance creation
# (`platform-capabilities | jq '.instance.open_ports'`); this used the free
# normal port container_port=10100 (public_port=8000 on this deployment) since
# vLLM's own 8000 was never one of the instance's allocated external ports.
# A *different* deployment will have different free ports — check first.
set -euo pipefail

python3 -c "
import yaml
d = yaml.safe_load(open('/etc/portal.yaml')) or {'applications': {}}
d['applications']['vLLM'] = {
    'hostname': 'localhost',
    'external_port': 10100,
    'internal_port': 8000,
    'open_path': '/v1/models',
    'name': 'vLLM',
}
yaml.safe_dump(d, open('/etc/portal.yaml', 'w'), sort_keys=False)
"

supervisorctl restart a reverse proxy

echo "Reachable at: http://\$HOST:\$PORT/v1/..."
echo "Auth: Authorization: Bearer \$QWEN_API_KEY  (this instance's portal token, see the target hardware instance page)"
