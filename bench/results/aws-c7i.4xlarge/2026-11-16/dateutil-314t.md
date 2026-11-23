608343 strings from 35 sources; 126638 ISO strings; {'python': '3.14.10', 'free_threaded': True, 'gil_enabled': False, 'platform': 'Linux-6.18.62-127.179.amzn2023.x86_64-x86_64-with-glibc2.36', 'machine': 'x86_64', 'cpu': 'Intel(R) Xeon(R) Platinum 8488C', 'cpus': 16, 'loadavg': (0.52294921875, 0.5615234375, 0.65283203125), 'python-dateutil': '2.9.0.post0', 'pyroid': '1.0.1', 'pyarrow': '26.0.1', 'pandas': None, 'ciso8601': None}
| call | us/string | vs dateutil |
| dateutil.parser.parse, loop                                |    26.110 |
| pyroid parse, loop                                         |     1.945 |    13.4x
| dateutil.parser.parse, 8 threads                           |     4.896 |     5.3x
| pyroid parse_batch, list, threads=1                        |     1.273 |    20.5x
| pyroid parse_batch, list, threads=8                        |     0.846 |    30.9x
| pyroid parse_batch, list, threads=None                     |     0.715 |    36.5x
| pyroid parse_batch, Arrow, threads=1                       |     0.725 |    36.0x
| pyroid parse_batch, Arrow, threads=8                       |     0.337 |    77.5x
| pyroid parse_batch, Arrow, threads=None                    |     0.268 |    97.5x
| ISO strings | us/string | vs dateutil isoparse |
| dateutil.parser.isoparse, loop                             |     3.442 |
| pyroid isoparse, loop                                      |     0.622 |     5.5x
| pyroid isoparse_batch, list, threads=1                     |     0.483 |     7.1x
| pyroid isoparse_batch, list, threads=None                  |     0.418 |     8.2x
| datetime.fromisoformat, loop (not dateutil's results)      |     0.164 |    21.0x
