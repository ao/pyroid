## json_repair benchmark (EC2 c8g.16xlarge devbox, shared, indicative)

Python 3.12.16; Linux-6.18.62-127.179.amzn2023.aarch64-aarch64-with-glibc2.36; 64 CPUs; json_repair 0.64.9, pyroid.json_repair 1.0.0. Load average before/after: [0.12, 0.15, 0.1] / [4.74, 2.33, 0.96]. Speedups compare against json_repair on the same interpreter.

### Single documents (best per call)

| doc | bytes | op | json_repair | pyroid.json_repair | speedup vs json_repair 0.64.9 |
|---|---:|---|---:|---:|---:|
| valid 1 KiB | 1,190 | repair_json | 12.6 us | 7.5 us | 1.7x |
| valid 1 KiB | 1,190 | loads | 6.6 us | 6.6 us | 1.0x |
| broken 1 KiB | 1,239 | repair_json | 366.6 us | 21.0 us | 17.5x |
| broken 1 KiB | 1,239 | loads | 358.4 us | 23.6 us | 15.2x |
| valid 1.0 MiB | 1,210,195 | repair_json | 10.07 ms | 7.64 ms | 1.3x |
| valid 1.0 MiB | 1,210,195 | loads | 5.74 ms | 5.75 ms | 1.0x |
| broken 1.0 MiB | 1,048,652 | repair_json | 231.99 ms | 16.00 ms | 14.5x |
| broken 1.0 MiB | 1,048,652 | loads | 227.78 ms | 17.56 ms | 13.0x |

### Pathological curves (best per call; > 20 s = killed; * = unchecked: json_repair did not reach this size)

| family | kind | bytes | json_repair | pyroid.json_repair | speedup vs json_repair 0.64.9 |
|---|---|---:|---:|---:|---:|
| truncated_string | realistic | 2,076 | 279.4 us | 37.2 us | 7.5x |
| truncated_string | realistic | 4,126 | 523.2 us | 68.1 us | 7.7x |
| truncated_string | realistic | 8,221 | 1.07 ms | 135.3 us | 7.9x |
| truncated_string | realistic | 16,416 | 2.09 ms | 276.1 us | 7.6x |
| truncated_string | realistic | 32,798 | 4.21 ms | 528.3 us | 8.0x |
| truncated_string | realistic | 65,566 | 8.38 ms | 1.14 ms | 7.4x |
| truncated_string | realistic | 131,104 | 16.33 ms | 2.42 ms | 6.8x |
| truncated_string | realistic | 262,176 | 33.45 ms | 4.70 ms | 7.1x |
| truncated_string | realistic | 524,318 | 65.17 ms | 9.89 ms | 6.6x |
| truncated_string | realistic | 1,048,612 | 132.31 ms | 19.57 ms | 6.8x |
| escaped_code_string | realistic | 2,117 | 982.6 us | 33.6 us | 29.2x |
| escaped_code_string | realistic | 4,165 | 1.89 ms | 61.9 us | 30.6x |
| escaped_code_string | realistic | 8,281 | 3.74 ms | 118.6 us | 31.5x |
| escaped_code_string | realistic | 16,455 | 7.29 ms | 231.8 us | 31.4x |
| escaped_code_string | realistic | 32,803 | 14.42 ms | 505.9 us | 28.5x |
| escaped_code_string | realistic | 65,633 | 28.63 ms | 983.7 us | 29.1x |
| escaped_code_string | realistic | 131,175 | 56.14 ms | 1.95 ms | 28.8x |
| escaped_code_string | realistic | 262,215 | 110.41 ms | 4.02 ms | 27.4x |
| escaped_code_string | realistic | 524,365 | 220.28 ms | 8.04 ms | 27.4x |
| escaped_code_string | realistic | 1,048,638 | 433.51 ms | 16.17 ms | 26.8x |
| escaped_code_string_skip_loads | realistic | 2,116 | 974.3 us | 29.2 us | 33.4x |
| escaped_code_string_skip_loads | realistic | 4,164 | 1.89 ms | 54.3 us | 34.8x |
| escaped_code_string_skip_loads | realistic | 8,280 | 3.71 ms | 104.8 us | 35.4x |
| escaped_code_string_skip_loads | realistic | 16,454 | 7.25 ms | 203.7 us | 35.6x |
| escaped_code_string_skip_loads | realistic | 32,802 | 14.31 ms | 457.5 us | 31.3x |
| escaped_code_string_skip_loads | realistic | 65,632 | 28.52 ms | 878.0 us | 32.5x |
| escaped_code_string_skip_loads | realistic | 131,174 | 55.46 ms | 1.75 ms | 31.6x |
| escaped_code_string_skip_loads | realistic | 262,214 | 111.57 ms | 3.55 ms | 31.4x |
| escaped_code_string_skip_loads | realistic | 524,364 | 219.83 ms | 7.14 ms | 30.8x |
| escaped_code_string_skip_loads | realistic | 1,048,637 | 429.61 ms | 14.37 ms | 29.9x |
| markdown_in_array | realistic | 2,129 | 657.4 us | 36.8 us | 17.8x |
| markdown_in_array | realistic | 4,259 | 1.30 ms | 69.1 us | 18.8x |
| markdown_in_array | realistic | 8,235 | 2.49 ms | 130.1 us | 19.1x |
| markdown_in_array | realistic | 16,535 | 4.97 ms | 255.2 us | 19.5x |
| markdown_in_array | realistic | 32,887 | 9.80 ms | 499.4 us | 19.6x |
| markdown_in_array | realistic | 65,591 | 19.48 ms | 1.08 ms | 18.1x |
| markdown_in_array | realistic | 131,145 | 38.73 ms | 2.13 ms | 18.2x |
| markdown_in_array | realistic | 262,299 | 76.33 ms | 4.33 ms | 17.6x |
| markdown_in_array | realistic | 524,349 | 151.80 ms | 8.82 ms | 17.2x |
| markdown_in_array | realistic | 1,048,749 | 304.08 ms | 17.61 ms | 17.3x |
| array_unescaped_quotes | realistic | 2,072 | 1.22 ms | 53.7 us | 22.8x |
| array_unescaped_quotes | realistic | 4,118 | 2.47 ms | 103.3 us | 23.9x |
| array_unescaped_quotes | realistic | 8,210 | 4.81 ms | 204.0 us | 23.6x |
| array_unescaped_quotes | realistic | 16,425 | 9.49 ms | 408.6 us | 23.2x |
| array_unescaped_quotes | realistic | 32,790 | 19.22 ms | 795.0 us | 24.2x |
| array_unescaped_quotes | realistic | 65,558 | 37.65 ms | 1.60 ms | 23.5x |
| array_unescaped_quotes | realistic | 131,094 | 73.90 ms | 3.35 ms | 22.1x |
| array_unescaped_quotes | realistic | 262,166 | 148.46 ms | 6.75 ms | 22.0x |
| array_unescaped_quotes | realistic | 524,327 | 296.15 ms | 12.93 ms | 22.9x |
| array_unescaped_quotes | realistic | 1,048,598 | 581.51 ms | 25.75 ms | 22.6x |
| single_quoted_apostrophes | realistic | 2,049 | 1.63 ms | 51.1 us | 31.9x |
| single_quoted_apostrophes | realistic | 4,106 | 3.16 ms | 99.0 us | 31.9x |
| single_quoted_apostrophes | realistic | 8,203 | 6.26 ms | 198.6 us | 31.5x |
| single_quoted_apostrophes | realistic | 16,397 | 12.41 ms | 389.3 us | 31.9x |
| single_quoted_apostrophes | realistic | 32,784 | 24.45 ms | 773.8 us | 31.6x |
| single_quoted_apostrophes | realistic | 65,544 | 48.05 ms | 1.53 ms | 31.3x |
| single_quoted_apostrophes | realistic | 131,082 | 95.68 ms | 3.02 ms | 31.7x |
| single_quoted_apostrophes | realistic | 262,148 | 193.83 ms | 6.26 ms | 31.0x |
| single_quoted_apostrophes | realistic | 524,291 | 375.19 ms | 13.53 ms | 27.7x |
| single_quoted_apostrophes | realistic | 1,048,577 | 751.79 ms | 25.86 ms | 29.1x |
| markdown_paren_list | realistic | 2,119 | 118.9 us | 10.0 us | 11.9x |
| markdown_paren_list | realistic | 4,158 | 223.1 us | 17.6 us | 12.7x |
| markdown_paren_list | realistic | 8,289 | 449.3 us | 33.4 us | 13.5x |
| markdown_paren_list | realistic | 16,463 | 891.9 us | 65.4 us | 13.6x |
| markdown_paren_list | realistic | 32,864 | 1.77 ms | 127.9 us | 13.8x |
| markdown_paren_list | realistic | 65,628 | 3.59 ms | 255.2 us | 14.1x |
| markdown_paren_list | realistic | 131,156 | 7.23 ms | 509.3 us | 14.2x |
| markdown_paren_list | realistic | 262,216 | 14.50 ms | 1.00 ms | 14.4x |
| markdown_paren_list | realistic | 524,351 | 29.38 ms | 2.03 ms | 14.5x |
| markdown_paren_list | realistic | 1,048,670 | 60.01 ms | 4.05 ms | 14.8x |
| records_missing_separators | realistic | 2,056 | 1.43 ms | 74.7 us | 19.2x |
| records_missing_separators | realistic | 4,104 | 2.84 ms | 144.5 us | 19.6x |
| records_missing_separators | realistic | 8,214 | 5.64 ms | 269.4 us | 20.9x |
| records_missing_separators | realistic | 16,404 | 11.49 ms | 523.4 us | 21.9x |
| records_missing_separators | realistic | 32,792 | 24.11 ms | 1.03 ms | 23.5x |
| records_missing_separators | realistic | 65,560 | 51.75 ms | 2.09 ms | 24.8x |
| records_missing_separators | realistic | 131,096 | 141.40 ms | 4.14 ms | 34.2x |
| records_missing_separators | realistic | 262,168 | 313.10 ms | 8.33 ms | 37.6x |
| records_missing_separators | realistic | 524,300 | 1.20 s | 16.67 ms | 72.2x |
| records_missing_separators | realistic | 1,048,580 | 5.12 s | 33.12 ms | 154.6x |
| table_rows_continued | realistic | 2,106 | 1.21 ms | 55.9 us | 21.7x |
| table_rows_continued | realistic | 4,158 | 2.34 ms | 99.7 us | 23.5x |
| table_rows_continued | realistic | 8,262 | 4.61 ms | 184.0 us | 25.0x |
| table_rows_continued | realistic | 16,452 | 9.08 ms | 355.2 us | 25.6x |
| table_rows_continued | realistic | 32,840 | 17.58 ms | 679.2 us | 25.9x |
| table_rows_continued | realistic | 65,600 | 34.28 ms | 1.33 ms | 25.7x |
| table_rows_continued | realistic | 131,140 | 67.92 ms | 2.60 ms | 26.2x |
| table_rows_continued | realistic | 262,214 | 133.50 ms | 5.04 ms | 26.5x |
| table_rows_continued | realistic | 524,344 | 260.38 ms | 10.03 ms | 26.0x |
| table_rows_continued | realistic | 1,048,648 | 519.33 ms | 19.69 ms | 26.4x |
| unbalanced_close_braces | adversarial | 2,058 | 2.12 ms | 50.4 us | 42.1x |
| unbalanced_close_braces | adversarial | 4,106 | 4.25 ms | 102.5 us | 41.5x |
| unbalanced_close_braces | adversarial | 8,202 | 8.45 ms | 186.7 us | 45.3x |
| unbalanced_close_braces | adversarial | 16,394 | 17.01 ms | 371.5 us | 45.8x |
| unbalanced_close_braces | adversarial | 32,778 | 34.24 ms | 734.5 us | 46.6x |
| unbalanced_close_braces | adversarial | 65,546 | 67.44 ms | 1.67 ms | 40.5x |
| unbalanced_close_braces | adversarial | 131,082 | 134.62 ms | 3.38 ms | 39.8x |
| unbalanced_close_braces | adversarial | 262,154 | 271.20 ms | 6.27 ms | 43.3x |
| unbalanced_close_braces | adversarial | 524,298 | 539.73 ms | 12.97 ms | 41.6x |
| unbalanced_close_braces | adversarial | 1,048,586 | 1.08 s | 27.47 ms | 39.4x |
| regex_class_quotes | adversarial | 2,062 | 99.61 ms | 52.6 us | 1894.3x |
| regex_class_quotes | adversarial | 4,110 | 389.66 ms | 99.1 us | 3931.7x |
| regex_class_quotes | adversarial | 8,206 | 1.54 s | 206.9 us | 7431.6x |
| regex_class_quotes | adversarial | 16,398 | 6.27 s | 382.8 us | 16366.3x |
| regex_class_quotes | adversarial | 32,782 | > 20 s | 786.9 us* | - |
| regex_class_quotes | adversarial | 65,550 | - | 1.59 ms* | - |
| regex_class_quotes | adversarial | 131,086 | - | 3.18 ms* | - |
| regex_class_quotes | adversarial | 262,158 | - | 6.34 ms* | - |
| regex_class_quotes | adversarial | 524,302 | - | 13.72 ms* | - |
| regex_class_quotes | adversarial | 1,048,590 | - | 25.82 ms* | - |
| nested_open_tuples | adversarial | 2,048 | 34.91 ms | 430.4 us | 81.1x |
| nested_open_tuples | adversarial | 4,094 | 70.88 ms | 496.6 us | 142.7x |
| nested_open_tuples | adversarial | 8,192 | 142.24 ms | 639.5 us | 222.4x |
| nested_open_tuples | adversarial | 16,382 | 285.50 ms | 926.4 us | 308.2x |
| nested_open_tuples | adversarial | 32,768 | 570.80 ms | 1.46 ms | 390.3x |
| nested_open_tuples | adversarial | 65,534 | 1.14 s | 2.53 ms | 451.0x |
| nested_open_tuples | adversarial | 131,072 | 2.28 s | 4.85 ms | 470.8x |
| nested_open_tuples | adversarial | 262,142 | 4.57 s | 9.44 ms | 483.5x |
| nested_open_tuples | adversarial | 524,288 | 9.15 s | 18.68 ms | 489.4x |
| nested_open_tuples | adversarial | 1,048,574 | 18.26 s | 35.88 ms | 508.8x |
| nested_open_containers | adversarial | 2,048 | 53.73 ms | 43.5 us | 1236.3x |
| nested_open_containers | adversarial | 4,096 | 109.66 ms | 75.9 us | 1444.8x |
| nested_open_containers | adversarial | 8,192 | 219.41 ms | 135.4 us | 1620.1x |
| nested_open_containers | adversarial | 16,384 | 441.23 ms | 262.9 us | 1678.1x |
| nested_open_containers | adversarial | 32,768 | 883.07 ms | 561.2 us | 1573.4x |
| nested_open_containers | adversarial | 65,536 | 1.77 s | 1.02 ms | 1735.9x |
| nested_open_containers | adversarial | 131,072 | 3.55 s | 2.39 ms | 1486.9x |
| nested_open_containers | adversarial | 262,144 | 7.05 s | 4.81 ms | 1466.5x |
| nested_open_containers | adversarial | 524,288 | 14.12 s | 9.80 ms | 1441.2x |
| nested_open_containers | adversarial | 1,048,576 | - | 20.13 ms* | - |
| valid_records | control | 2,608 | 24.8 us | 18.5 us | 1.3x |
| valid_records | control | 5,024 | 43.9 us | 35.2 us | 1.2x |
| valid_records | control | 9,549 | 77.9 us | 64.8 us | 1.2x |
| valid_records | control | 18,953 | 154.3 us | 130.6 us | 1.2x |
| valid_records | control | 37,962 | 307.4 us | 254.4 us | 1.2x |
| valid_records | control | 75,965 | 606.8 us | 489.8 us | 1.2x |
| valid_records | control | 151,864 | 1.25 ms | 965.2 us | 1.3x |
| valid_records | control | 302,895 | 2.50 ms | 1.90 ms | 1.3x |
| valid_records | control | 604,989 | 4.92 ms | 3.74 ms | 1.3x |
| valid_records | control | 1,209,710 | 9.88 ms | 7.43 ms | 1.3x |
| valid_records_skip_loads | control | 2,608 | 577.6 us | 40.5 us | 14.3x |
| valid_records_skip_loads | control | 5,024 | 1.08 ms | 76.7 us | 14.0x |
| valid_records_skip_loads | control | 9,549 | 1.99 ms | 137.6 us | 14.5x |
| valid_records_skip_loads | control | 18,953 | 4.05 ms | 270.7 us | 15.0x |
| valid_records_skip_loads | control | 37,962 | 8.15 ms | 549.4 us | 14.8x |
| valid_records_skip_loads | control | 75,965 | 16.11 ms | 1.05 ms | 15.3x |
| valid_records_skip_loads | control | 151,864 | 32.30 ms | 2.11 ms | 15.3x |
| valid_records_skip_loads | control | 302,895 | 64.59 ms | 4.20 ms | 15.4x |
| valid_records_skip_loads | control | 604,989 | 127.12 ms | 8.28 ms | 15.3x |
| valid_records_skip_loads | control | 1,209,710 | 257.85 ms | 16.49 ms | 15.6x |
| broken_records | control | 2,060 | 452.3 us | 29.8 us | 15.2x |
| broken_records | control | 4,169 | 952.0 us | 60.0 us | 15.9x |
| broken_records | control | 8,423 | 1.92 ms | 129.1 us | 14.9x |
| broken_records | control | 16,478 | 3.71 ms | 256.5 us | 14.5x |
| broken_records | control | 32,851 | 7.48 ms | 516.9 us | 14.5x |
| broken_records | control | 65,613 | 14.78 ms | 1.00 ms | 14.8x |
| broken_records | control | 131,040 | 29.00 ms | 2.01 ms | 14.4x |
| broken_records | control | 262,134 | 57.37 ms | 4.04 ms | 14.2x |
| broken_records | control | 524,574 | 115.25 ms | 8.05 ms | 14.3x |
| broken_records | control | 1,048,657 | 231.95 ms | 15.84 ms | 14.6x |
| prose_fragments | control | 2,068 | 758.6 us | 32.1 us | 23.6x |
| prose_fragments | control | 4,098 | 1.52 ms | 59.8 us | 25.4x |
| prose_fragments | control | 8,240 | 3.05 ms | 116.6 us | 26.1x |
| prose_fragments | control | 16,400 | 6.00 ms | 228.5 us | 26.2x |
| prose_fragments | control | 32,780 | 12.05 ms | 449.6 us | 26.8x |
| prose_fragments | control | 65,546 | 23.82 ms | 895.1 us | 26.6x |
| prose_fragments | control | 131,080 | 46.93 ms | 1.77 ms | 26.6x |
| prose_fragments | control | 262,148 | 93.57 ms | 3.50 ms | 26.7x |
| prose_fragments | control | 524,346 | 187.62 ms | 7.00 ms | 26.8x |
| prose_fragments | control | 1,048,628 | 367.80 ms | 13.90 ms | 26.5x |

### Time exponents (log-log slope of best time vs bytes over the points >= 1 ms; in brackets over the 3 largest sizes)

| family | kind | json_repair | pyroid.json_repair |
|---|---|---:|---:|
| truncated_string | realistic | 0.99 (0.99) | 1.02 (1.03) |
| escaped_code_string | realistic | 0.98 (0.99) | 1.02 (1.00) |
| escaped_code_string_skip_loads | realistic | 0.98 (0.97) | 1.01 (1.01) |
| markdown_in_array | realistic | 0.99 (1.00) | 1.01 (1.01) |
| array_unescaped_quotes | realistic | 0.99 (0.98) | 1.00 (0.97) |
| single_quoted_apostrophes | realistic | 0.98 (0.98) | 1.03 (1.02) |
| markdown_paren_list | realistic | 1.02 (1.02) | 1.01 (1.01) |
| records_missing_separators | realistic | 1.26 (2.02) | 1.00 (1.00) |
| table_rows_continued | realistic | 0.97 (0.98) | 0.97 (0.98) |
| unbalanced_close_braces | adversarial | 1.00 (1.00) | 1.00 (1.07) |
| regex_class_quotes | adversarial | 2.00 (2.01) | 1.02 (1.01) |
| nested_open_tuples | adversarial | 1.00 (1.00) | 0.93 (0.96) |
| nested_open_containers | adversarial | 1.00 (1.00) | 1.07 (1.03) |
| valid_records | control | 0.99 (0.99) | 0.98 (0.98) |
| valid_records_skip_loads | control | 1.00 (1.00) | 0.99 (0.99) |
| broken_records | control | 0.99 (1.01) | 1.00 (0.99) |
| prose_fragments | control | 0.99 (0.99) | 0.99 (0.99) |

