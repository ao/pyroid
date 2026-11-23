
## Small queries, one describe_instances response (µs per call)

| Case | Expression | jmespath | pyroid | Speedup |
|---|---|---:|---:|---:|
| field, search() | `Reservations` | 1.62 | 0.19 | 8.4x |
| field, compiled | `Reservations` | 1.25 | 0.11 | 11.3x |
| path, search() | `Reservations[0].Instances[0].State.Name` | 4.57 | 0.29 | 16.0x |
| path, compiled | `Reservations[0].Instances[0].State.Name` | 4.19 | 0.20 | 21.4x |
| projection, search() | `Reservations[].Instances[].InstanceId` | 4.66 | 0.53 | 8.8x |
| projection, compiled | `Reservations[].Instances[].InstanceId` | 4.26 | 0.44 | 9.8x |
| filter, search() | `Reservations[].Instances[?State.Name=='running'].InstanceId[]` | 12.90 | 0.84 | 15.4x |
| filter, compiled | `Reservations[].Instances[?State.Name=='running'].InstanceId[]` | 12.47 | 0.72 | 17.3x |
| multi-select, search() | `Reservations[].Instances[].{id: InstanceId, type: InstanceType}` | 8.40 | 0.73 | 11.6x |
| multi-select, compiled | `Reservations[].Instances[].{id: InstanceId, type: InstanceType}` | 7.98 | 0.61 | 13.0x |
| tag lookup, search() | `Reservations[].Instances[].Tags[?Key=='Name'].Value[]` | 18.12 | 1.33 | 13.6x |
| tag lookup, compiled | `Reservations[].Instances[].Tags[?Key=='Name'].Value[]` | 17.69 | 1.23 | 14.3x |
| sort_by, search() | `sort_by(Reservations[].Instances[], &LaunchTime)[-1].InstanceId` | 12.49 | 0.88 | 14.2x |
| sort_by, compiled | `sort_by(Reservations[].Instances[], &LaunchTime)[-1].InstanceId` | 12.04 | 0.77 | 15.6x |
| join, search() | `join(', ', Reservations[].Instances[].InstanceId)` | 7.68 | 0.84 | 9.2x |
| join, compiled | `join(', ', Reservations[].Instances[].InstanceId)` | 7.30 | 0.71 | 10.3x |

## A new expression every call: parse and search (µs per call)

| Case | Expression | jmespath | pyroid | Speedup |
|---|---|---:|---:|---:|
| parse + search | `Reservations[].Instances[?State.Name=='running' && Tags[?Value=='web-{0}']].InstanceId` | 52.33 | 5.64 | 9.3x |

## 10,000 records (ms per query)

| Case | Expression | jmespath | pyroid | Speedup |
|---|---|---:|---:|---:|
| projection | `[*].address.city` | 7.89 | 0.39 | 20.4x |
| filter | `[?age > `30`].name` | 16.82 | 0.69 | 24.3x |
| filter, and | `[?age > `30` && address.city == 'Oslo'].id` | 34.24 | 1.40 | 24.4x |
| multi-select | `[*].{n: name, c: address.city}` | 18.47 | 1.24 | 14.9x |
| sort_by | `sort_by(@, &age)[0].id` | 6.83 | 0.46 | 15.0x |
| max_by | `max_by(@, &score).id` | 6.13 | 0.44 | 13.9x |
| sum | `sum([*].age)` | 3.43 | 0.36 | 9.4x |
| contains | `[?contains(tags, 'a')] | length(@)` | 21.78 | 1.13 | 19.2x |

## 1,000,000 records (ms per query)

| Case | Expression | jmespath | pyroid | Speedup |
|---|---|---:|---:|---:|
| projection | `[*].address.city` | 775.80 | 44.37 | 17.5x |
| filter | `[?age > `30`].name` | 1,673.78 | 76.58 | 21.9x |
| filter, and | `[?age > `30` && address.city == 'Oslo'].id` | 3,447.84 | 149.49 | 23.1x |
| multi-select | `[*].{n: name, c: address.city}` | 1,912.92 | 179.45 | 10.7x |
| sort_by | `sort_by(@, &age)[0].id` | 722.05 | 77.62 | 9.3x |
| max_by | `max_by(@, &score).id` | 620.08 | 48.37 | 12.8x |
| sum | `sum([*].age)` | 347.97 | 43.56 | 8.0x |
| contains | `[?contains(tags, 'a')] | length(@)` | 2,190.00 | 120.10 | 18.2x |
