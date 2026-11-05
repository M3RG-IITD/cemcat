# Llamat prompt variants

Generation is capped at the length of one label. A variant is usable when every row parses.

| model | prompt | max tokens | precision | recall | f1 | parsed | unparsed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llamat-2-chat | yesno | 2 | 0.786 | 0.239 | 0.367 | 36 | 24 |
| llamat-2-chat | dict | 12 | 1.000 | 0.022 | 0.043 | 1 | 59 |
| llamat-2-chat | json | 12 | 0.000 | 0.000 | 0.000 | 0 | 60 |
| llamat-3-chat | yesno | 2 | 0.000 | 0.000 | 0.000 | 55 | 5 |
| llamat-3-chat | dict | 12 | 0.000 | 0.000 | 0.000 | 37 | 23 |
| llamat-3-chat | json | 12 | 0.000 | 0.000 | 0.000 | 30 | 30 |
