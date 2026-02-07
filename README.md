# solar-pointer
Point at anything in the solar system

## Quick start
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py --name "ISS (ZARYA)"
```

Use a NORAD catalog number instead:
```
python3 main.py --catnr 25544
```

## Persistent calibration config
Create/update the persistent config by telling the software what the current physical pointing is.

Example: if the pointer is currently due east at horizon:
```
python3 main.py --set-reference --reference-az-deg 90 --reference-alt-deg 0
```

Manual calibration mode for north/horizon:
```
python3 main.py --calibrate-north-horizon
```
This mode disables torque so you can move the pointer by hand, then saves that pose as `az=0`, `alt=0`.

This writes `pointer_config.json` with the current encoder ticks mapped to az/el degrees.

Print resolved config:
```
python3 main.py --show-config
```

One-shot test in degree space:
```
python3 main.py --point-az-deg 0 --point-alt-deg 0
```

When servo output is enabled, startup behavior is:
1. Point to north/horizon (`az=0`, `alt=0`)
2. Hold for 3 seconds
3. Point to the requested target (one-shot target or live ISS track)

Override hold time if needed:
```
python3 main.py --name "ISS (ZARYA)" --startup-hold-seconds 1.5
```

## Track ISS + map animation
Run continuous tracking (defaults to Auckland observer):
```
python3 main.py \
  --name "ISS (ZARYA)" \
  --config pointer_config.json
```

Useful runtime overrides:
```
python3 main.py --name "ISS (ZARYA)" --az-min-deg 0 --az-max-deg 90
```
