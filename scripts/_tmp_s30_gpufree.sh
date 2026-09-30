#!/usr/bin/env bash
# Free GPUs per SKU outside the excluded / reserved nodes (read-only).
EXCL="ee-l40s-01 ee-l40s-02 cs-4090-09 ise-6000p-01 ise-6000p-02 ise-6000p-03 ise-6000p-04 ise-6000p-05 ise-6000p-06 ise-6000p-07"
RES=$(scontrol show reservation root_20 2>/dev/null | grep -oE "Nodes=[^ ]+" | head -1 | cut -d= -f2)
RESX=$(scontrol show hostnames "$RES" 2>/dev/null | tr '\n' ' ')
scontrol show nodes -o 2>/dev/null | python3 -c "
import re, sys
excl = set('$EXCL'.split()) | set('$RESX'.split())
free, tot = {}, {}
for rec in sys.stdin:
    kv = dict(p.split('=', 1) for p in rec.split() if '=' in p)
    name, state = kv.get('NodeName', ''), kv.get('State', '').upper()
    for sku, n in re.findall(r'gres/gpu:([a-z0-9_]+)=(\d+)', kv.get('CfgTRES', '')):
        tot[sku] = tot.get(sku, 0) + int(n)
        if name in excl or any(t in state for t in ('DOWN', 'DRAIN', 'NOT_RESPONDING', 'FAIL', 'MAINT', 'RESERVED')):
            continue
        a = re.search(r'gres/gpu:' + sku + r'=(\d+)', kv.get('AllocTRES', ''))
        f = int(n) - int(a.group(1) if a else 0)
        if f > 0:
            free.setdefault(sku, []).append(f'{name}:{f}')
for sku in sorted(tot):
    print(f'{sku:14s} total {tot[sku]:3d} | free now: {sum(int(x.split(\":\")[1]) for x in free.get(sku, []))}  {\" \".join(free.get(sku, [])[:8])}')
"
scontrol show node cs-pheno-06 2>/dev/null | grep -oE "Gres=[^ ]+|CfgTRES=[^ ]+" | head -2
