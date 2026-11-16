## Small queries, one describe_instances response (µs per call)

| Case | Expression | jmespath | pyroid | Speedup |
|---|---|---:|---:|---:|
| field, search() | `Reservations` | 1.49 | 0.15 | 10.0x |
| field, compiled | `Reservations` | 1.23 | 0.24 | 5.1x |
| path, search() | `Reservations[0].Instances[0].State.Name` | 4.44 | 0.22 | 20.1x |
| path, compiled | `Reservations[0].Instances[0].State.Name` | 4.02 | 0.16 | 25.5x |
| projection, search() | `Reservations[].Instances[].InstanceId` | 4.49 | 0.46 | 9.7x |
| projection, compiled | `Reservations[].Instances[].InstanceId` | 4.14 | 0.40 | 10.2x |
| filter, search() | `Reservations[].Instances[?State.Name=='running'].InstanceId[]` | 12.77 | 0.79 | 16.2x |
| filter, compiled | `Reservations[].Instances[?State.Name=='running'].InstanceId[]` | 12.30 | 0.72 | 17.2x |
| multi-select, search() | `Reservations[].Instances[].{id: InstanceId, type: InstanceType}` | 8.10 | 0.63 | 12.8x |
| multi-select, compiled | `Reservations[].Instances[].{id: InstanceId, type: InstanceType}` | 7.77 | 0.57 | 13.7x |
| tag lookup, search() | `Reservations[].Instances[].Tags[?Key=='Name'].Value[]` | 17.64 | 1.34 | 13.2x |
| tag lookup, compiled | `Reservations[].Instances[].Tags[?Key=='Name'].Value[]` | 17.28 | 1.26 | 13.7x |
| sort_by, search() | `sort_by(Reservations[].Instances[], &LaunchTime)[-1].InstanceId` | 12.86 | 0.82 | 15.6x |
| sort_by, compiled | `sort_by(Reservations[].Instances[], &LaunchTime)[-1].InstanceId` | 12.37 | 0.76 | 16.3x |
| join, search() | `join(', ', Reservations[].Instances[].InstanceId)` | 7.90 | 0.73 | 10.8x |
| join, compiled | `join(', ', Reservations[].Instances[].InstanceId)` | 7.34 | 0.68 | 10.9x |

## A new expression every call: parse and search (µs per call)

| Case | Expression | jmespath | pyroid | Speedup |
|---|---|---:|---:|---:|
| parse + search | `Reservations[].Instances[?State.Name=='running' && Tags[?Value=='web-{0}']].InstanceId` | 55.78 | 5.61 | 9.9x |

## 10,000 records (ms per query)

| Case | Expression | jmespath | pyroid | Speedup |
|---|---|---:|---:|---:|
| projection | `[*].address.city` | 6.71 | 0.44 | 15.2x |
| filter | `[?age > `30`].name` | 15.76 | 0.85 | 18.5x |
| filter, and | `[?age > `30` && address.city == 'Oslo'].id` | 31.14 | 1.53 | 20.3x |
| multi-select | `[*].{n: name, c: address.city}` | 15.79 | 1.28 | 12.3x |
| sort_by | `sort_by(@, &age)[0].id` | 6.20 | 0.50 | 12.5x |
| max_by | `max_by(@, &score).id` | 5.43 | 0.47 | 11.6x |
| sum | `sum([*].age)` | 2.82 | 0.38 | 7.4x |
| contains | `[?contains(tags, 'a')] | length(@)` | 19.28 | 1.24 | 15.5x |

## 1,000,000 records (ms per query)

| Case | Expression | jmespath | pyroid | Speedup |
|---|---|---:|---:|---:|
| projection | `[*].address.city` | 682.74 | 87.38 | 7.8x |
| filter | `[?age > `30`].name` | 1,671.97 | 109.16 | 15.3x |
| filter, and | `[?age > `30` && address.city == 'Oslo'].id` | 3,157.32 | 180.80 | 17.5x |
| multi-select | `[*].{n: name, c: address.city}` | 1,722.95 | 272.26 | 6.3x |
| sort_by | `sort_by(@, &age)[0].id` | 790.32 | 137.11 | 5.8x |
| max_by | `max_by(@, &score).id` | 608.69 | 58.33 | 10.4x |
| sum | `sum([*].age)` | 289.59 | 68.74 | 4.2x |
| contains | `[?contains(tags, 'a')] | length(@)` | 1,968.25 | 167.78 | 11.7x |
