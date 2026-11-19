## Small queries, one describe_instances response (µs per call)

| Case | Expression | jmespath | pyroid | Speedup |
|---|---|---:|---:|---:|
| field, search() | `Reservations` | 1.48 | 0.14 | 10.3x |
| field, compiled | `Reservations` | 1.20 | 0.09 | 12.8x |
| path, search() | `Reservations[0].Instances[0].State.Name` | 4.46 | 0.21 | 21.3x |
| path, compiled | `Reservations[0].Instances[0].State.Name` | 4.03 | 0.15 | 26.0x |
| projection, search() | `Reservations[].Instances[].InstanceId` | 4.54 | 0.46 | 9.8x |
| projection, compiled | `Reservations[].Instances[].InstanceId` | 4.12 | 0.40 | 10.2x |
| filter, search() | `Reservations[].Instances[?State.Name=='running'].InstanceId[]` | 12.63 | 0.78 | 16.2x |
| filter, compiled | `Reservations[].Instances[?State.Name=='running'].InstanceId[]` | 12.15 | 0.72 | 16.9x |
| multi-select, search() | `Reservations[].Instances[].{id: InstanceId, type: InstanceType}` | 8.02 | 0.62 | 12.8x |
| multi-select, compiled | `Reservations[].Instances[].{id: InstanceId, type: InstanceType}` | 7.53 | 0.57 | 13.3x |
| tag lookup, search() | `Reservations[].Instances[].Tags[?Key=='Name'].Value[]` | 17.31 | 1.34 | 13.0x |
| tag lookup, compiled | `Reservations[].Instances[].Tags[?Key=='Name'].Value[]` | 16.72 | 1.26 | 13.2x |
| sort_by, search() | `sort_by(Reservations[].Instances[], &LaunchTime)[-1].InstanceId` | 12.75 | 0.82 | 15.5x |
| sort_by, compiled | `sort_by(Reservations[].Instances[], &LaunchTime)[-1].InstanceId` | 12.09 | 0.76 | 15.9x |
| join, search() | `join(', ', Reservations[].Instances[].InstanceId)` | 7.65 | 0.73 | 10.4x |
| join, compiled | `join(', ', Reservations[].Instances[].InstanceId)` | 7.17 | 0.67 | 10.7x |

## A new expression every call: parse and search (µs per call)

| Case | Expression | jmespath | pyroid | Speedup |
|---|---|---:|---:|---:|
| parse + search | `Reservations[].Instances[?State.Name=='running' && Tags[?Value=='web-{0}']].InstanceId` | 55.30 | 5.46 | 10.1x |

## 10,000 records (ms per query)

| Case | Expression | jmespath | pyroid | Speedup |
|---|---|---:|---:|---:|
| projection | `[*].address.city` | 6.71 | 0.45 | 15.0x |
| filter | `[?age > `30`].name` | 15.89 | 0.85 | 18.7x |
| filter, and | `[?age > `30` && address.city == 'Oslo'].id` | 30.90 | 1.54 | 20.1x |
| multi-select | `[*].{n: name, c: address.city}` | 15.86 | 1.29 | 12.3x |
| sort_by | `sort_by(@, &age)[0].id` | 6.32 | 0.50 | 12.7x |
| max_by | `max_by(@, &score).id` | 5.49 | 0.47 | 11.7x |
| sum | `sum([*].age)` | 2.80 | 0.38 | 7.4x |
| contains | `[?contains(tags, 'a')] | length(@)` | 18.93 | 1.24 | 15.2x |

## 1,000,000 records (ms per query)

| Case | Expression | jmespath | pyroid | Speedup |
|---|---|---:|---:|---:|
| projection | `[*].address.city` | 687.05 | 93.39 | 7.4x |
| filter | `[?age > `30`].name` | 1,687.21 | 110.97 | 15.2x |
| filter, and | `[?age > `30` && address.city == 'Oslo'].id` | 3,116.75 | 180.62 | 17.3x |
| multi-select | `[*].{n: name, c: address.city}` | 1,698.96 | 269.79 | 6.3x |
| sort_by | `sort_by(@, &age)[0].id` | 789.21 | 140.44 | 5.6x |
| max_by | `max_by(@, &score).id` | 610.66 | 60.92 | 10.0x |
| sum | `sum([*].age)` | 287.48 | 69.22 | 4.2x |
| contains | `[?contains(tags, 'a')] | length(@)` | 1,928.85 | 163.92 | 11.8x |
