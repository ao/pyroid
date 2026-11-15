
## Small queries, one describe_instances response (µs per call)

| Case | Expression | jmespath | pyroid | Speedup |
|---|---|---:|---:|---:|
| field, search() | `Reservations` | 1.47 | 0.14 | 10.1x |
| field, compiled | `Reservations` | 1.19 | 0.09 | 12.7x |
| path, search() | `Reservations[0].Instances[0].State.Name` | 4.34 | 0.22 | 19.4x |
| path, compiled | `Reservations[0].Instances[0].State.Name` | 3.99 | 0.17 | 23.4x |
| projection, search() | `Reservations[].Instances[].InstanceId` | 4.41 | 0.45 | 9.7x |
| projection, compiled | `Reservations[].Instances[].InstanceId` | 4.04 | 0.40 | 10.0x |
| filter, search() | `Reservations[].Instances[?State.Name=='running'].InstanceId[]` | 12.52 | 0.78 | 16.1x |
| filter, compiled | `Reservations[].Instances[?State.Name=='running'].InstanceId[]` | 11.97 | 0.72 | 16.5x |
| multi-select, search() | `Reservations[].Instances[].{id: InstanceId, type: InstanceType}` | 7.86 | 0.63 | 12.4x |
| multi-select, compiled | `Reservations[].Instances[].{id: InstanceId, type: InstanceType}` | 7.49 | 0.58 | 13.0x |
| tag lookup, search() | `Reservations[].Instances[].Tags[?Key=='Name'].Value[]` | 17.61 | 1.32 | 13.3x |
| tag lookup, compiled | `Reservations[].Instances[].Tags[?Key=='Name'].Value[]` | 16.94 | 1.25 | 13.5x |
| sort_by, search() | `sort_by(Reservations[].Instances[], &LaunchTime)[-1].InstanceId` | 12.23 | 0.83 | 14.8x |
| sort_by, compiled | `sort_by(Reservations[].Instances[], &LaunchTime)[-1].InstanceId` | 11.77 | 0.76 | 15.4x |
| join, search() | `join(', ', Reservations[].Instances[].InstanceId)` | 7.55 | 0.75 | 10.1x |
| join, compiled | `join(', ', Reservations[].Instances[].InstanceId)` | 7.02 | 0.68 | 10.3x |

## A new expression every call: parse and search (µs per call)

| Case | Expression | jmespath | pyroid | Speedup |
|---|---|---:|---:|---:|
| parse + search | `Reservations[].Instances[?State.Name=='running' && Tags[?Value=='web-{0}']].InstanceId` | 53.49 | 5.57 | 9.6x |

## 10,000 records (ms per query)

| Case | Expression | jmespath | pyroid | Speedup |
|---|---|---:|---:|---:|
| projection | `[*].address.city` | 6.51 | 0.47 | 13.7x |
| filter | `[?age > `30`].name` | 15.70 | 0.84 | 18.6x |
| filter, and | `[?age > `30` && address.city == 'Oslo'].id` | 31.15 | 1.55 | 20.1x |
| multi-select | `[*].{n: name, c: address.city}` | 15.63 | 1.31 | 12.0x |
| sort_by | `sort_by(@, &age)[0].id` | 6.16 | 0.50 | 12.4x |
| max_by | `max_by(@, &score).id` | 5.36 | 0.49 | 11.0x |
| sum | `sum([*].age)` | 2.81 | 0.44 | 6.4x |
| contains | `[?contains(tags, 'a')] | length(@)` | 18.91 | 1.24 | 15.2x |

## 1,000,000 records (ms per query)

| Case | Expression | jmespath | pyroid | Speedup |
|---|---|---:|---:|---:|
| projection | `[*].address.city` | 686.80 | 93.52 | 7.3x |
| filter | `[?age > `30`].name` | 1,681.10 | 108.00 | 15.6x |
| filter, and | `[?age > `30` && address.city == 'Oslo'].id` | 3,149.44 | 184.20 | 17.1x |
| multi-select | `[*].{n: name, c: address.city}` | 1,714.55 | 271.36 | 6.3x |
| sort_by | `sort_by(@, &age)[0].id` | 776.99 | 128.30 | 6.1x |
| max_by | `max_by(@, &score).id` | 586.41 | 60.69 | 9.7x |
| sum | `sum([*].age)` | 287.94 | 71.62 | 4.0x |
| contains | `[?contains(tags, 'a')] | length(@)` | 1,934.20 | 163.32 | 11.8x |
