
## Small queries, one describe_instances response (µs per call)

| Case | Expression | jmespath | pyroid | Speedup |
|---|---|---:|---:|---:|
| field, search() | `Reservations` | 1.51 | 0.15 | 10.3x |
| field, compiled | `Reservations` | 1.24 | 0.19 | 6.6x |
| path, search() | `Reservations[0].Instances[0].State.Name` | 4.51 | 0.23 | 19.6x |
| path, compiled | `Reservations[0].Instances[0].State.Name` | 4.17 | 0.17 | 24.5x |
| projection, search() | `Reservations[].Instances[].InstanceId` | 4.48 | 0.46 | 9.7x |
| projection, compiled | `Reservations[].Instances[].InstanceId` | 4.17 | 0.40 | 10.4x |
| filter, search() | `Reservations[].Instances[?State.Name=='running'].InstanceId[]` | 12.71 | 0.77 | 16.4x |
| filter, compiled | `Reservations[].Instances[?State.Name=='running'].InstanceId[]` | 12.23 | 0.72 | 17.1x |
| multi-select, search() | `Reservations[].Instances[].{id: InstanceId, type: InstanceType}` | 8.25 | 0.64 | 12.9x |
| multi-select, compiled | `Reservations[].Instances[].{id: InstanceId, type: InstanceType}` | 7.82 | 0.58 | 13.5x |
| tag lookup, search() | `Reservations[].Instances[].Tags[?Key=='Name'].Value[]` | 17.64 | 1.33 | 13.2x |
| tag lookup, compiled | `Reservations[].Instances[].Tags[?Key=='Name'].Value[]` | 17.06 | 1.26 | 13.6x |
| sort_by, search() | `sort_by(Reservations[].Instances[], &LaunchTime)[-1].InstanceId` | 12.67 | 0.83 | 15.3x |
| sort_by, compiled | `sort_by(Reservations[].Instances[], &LaunchTime)[-1].InstanceId` | 12.27 | 0.76 | 16.1x |
| join, search() | `join(', ', Reservations[].Instances[].InstanceId)` | 7.81 | 0.75 | 10.5x |
| join, compiled | `join(', ', Reservations[].Instances[].InstanceId)` | 7.40 | 0.68 | 10.9x |

## A new expression every call: parse and search (µs per call)

| Case | Expression | jmespath | pyroid | Speedup |
|---|---|---:|---:|---:|
| parse + search | `Reservations[].Instances[?State.Name=='running' && Tags[?Value=='web-{0}']].InstanceId` | 53.49 | 5.58 | 9.6x |

## 10,000 records (ms per query)

| Case | Expression | jmespath | pyroid | Speedup |
|---|---|---:|---:|---:|
| projection | `[*].address.city` | 6.62 | 0.52 | 12.7x |
| filter | `[?age > `30`].name` | 15.74 | 0.84 | 18.7x |
| filter, and | `[?age > `30` && address.city == 'Oslo'].id` | 30.73 | 1.54 | 20.0x |
| multi-select | `[*].{n: name, c: address.city}` | 15.82 | 1.34 | 11.8x |
| sort_by | `sort_by(@, &age)[0].id` | 6.23 | 0.50 | 12.6x |
| max_by | `max_by(@, &score).id` | 5.43 | 0.49 | 11.2x |
| sum | `sum([*].age)` | 2.84 | 0.45 | 6.4x |
| contains | `[?contains(tags, 'a')] | length(@)` | 18.81 | 1.24 | 15.2x |

## 1,000,000 records (ms per query)

| Case | Expression | jmespath | pyroid | Speedup |
|---|---|---:|---:|---:|
| projection | `[*].address.city` | 683.58 | 97.62 | 7.0x |
| filter | `[?age > `30`].name` | 1,690.89 | 108.26 | 15.6x |
| filter, and | `[?age > `30` && address.city == 'Oslo'].id` | 3,140.39 | 188.78 | 16.6x |
| multi-select | `[*].{n: name, c: address.city}` | 1,710.86 | 271.37 | 6.3x |
| sort_by | `sort_by(@, &age)[0].id` | 767.91 | 127.62 | 6.0x |
| max_by | `max_by(@, &score).id` | 585.47 | 60.03 | 9.8x |
| sum | `sum([*].age)` | 290.66 | 71.92 | 4.0x |
| contains | `[?contains(tags, 'a')] | length(@)` | 1,946.58 | 163.58 | 11.9x |
