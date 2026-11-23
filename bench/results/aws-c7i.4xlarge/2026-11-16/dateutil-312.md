608343 strings from 35 sources; 139939 ISO strings; {'python': '3.12.16', 'free_threaded': False, 'gil_enabled': True, 'platform': 'Linux-6.18.62-127.179.amzn2023.x86_64-x86_64-with-glibc2.36', 'machine': 'x86_64', 'cpu': 'Intel(R) Xeon(R) Platinum 8488C', 'cpus': 16, 'loadavg': (0.25439453125, 0.3916015625, 0.61962890625), 'python-dateutil': '2.9.0.post0', 'pyroid': '1.0.1', 'pyarrow': '26.0.1', 'pandas': '3.1.1', 'ciso8601': '2.4.1'}
| call | us/string | vs dateutil |
| dateutil.parser.parse, loop                                |    26.988 |
| pyroid parse, loop                                         |     1.757 |    15.4x
| pyroid parse_batch, list, threads=1                        |     1.192 |    22.6x
| pyroid parse_batch, list, threads=8                        |     0.760 |    35.5x
| pyroid parse_batch, list, threads=None                     |     0.598 |    45.1x
| pyroid parse_batch, Arrow, threads=1                       |     0.727 |    37.1x
| pyroid parse_batch, Arrow, threads=8                       |     0.333 |    81.1x
| pyroid parse_batch, Arrow, threads=None                    |     0.268 |   100.8x
| pandas.to_datetime(format='mixed') (not dateutil's results) |    29.074 |     0.9x
| pandas.to_datetime(format=None) (not dateutil's results)   |     0.747 |    36.1x
| ISO strings | us/string | vs dateutil isoparse |
| dateutil.parser.isoparse, loop                             |     3.504 |
| pyroid isoparse, loop                                      |     0.971 |     3.6x
| pyroid isoparse_batch, list, threads=1                     |     0.949 |     3.7x
| pyroid isoparse_batch, list, threads=None                  |     0.842 |     4.2x
| datetime.fromisoformat, loop (not dateutil's results)      |     0.129 |    27.3x
| ciso8601.parse_datetime, loop (not dateutil's results)     |     0.165 |    21.3x
