# solar-pointer
Point at satellites, planets, interplanetary spacecraft, stars, constellations, and curated deep-sky targets.

## Quick start
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py --target "ISS (ZARYA)" --target-kind satellite
```

Search before tracking:
```
python3 main.py --search voyager --target-kind spacecraft
python3 main.py --search sirius --target-kind star
python3 main.py --search orion --target-kind constellation
```

Track common target classes:
```
python3 main.py --target Mars --target-kind solar-system
python3 main.py --target "Voyager 1" --target-kind spacecraft
python3 main.py --target Sirius --target-kind star
python3 main.py --target Orion --target-kind constellation
python3 main.py --target M31 --target-kind dso
```

Deprecated satellite aliases are still supported:
```
python3 main.py --name "ISS (ZARYA)"
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
  --target "ISS (ZARYA)" \
  --target-kind satellite \
  --config pointer_config.json
```

Useful runtime overrides:
```
python3 main.py --target Mars --target-kind solar-system --az-min-deg 0 --az-max-deg 90
```

Earth satellites render on an Earth ground-track map. All other targets render in an azimuth/altitude sky view.

## Web control
Run the local web interface:
```
python3 main.py --serve
```

By default it starts at:
```
http://127.0.0.1:8765
```
