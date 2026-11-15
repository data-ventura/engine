# dataventura

Connect your own Python code to [Data Ventura](https://dataventura.com). Functions you decorate with `@dv.transform`, `@dv.view` or `@dv.reader` appear in the app's menus and run in your own environment, with data passed through Arrow shared memory.

```python
import polars as pl
import dataventura as dv

@dv.transform(columns=["fare", "distance"])
def fare_per_km(df: pl.DataFrame) -> pl.DataFrame:
    return df.select((pl.col("fare") / pl.col("distance")).alias("fare_per_km"))
```

The type hint on the first parameter chooses the input: `pyarrow.Table` (the default), `polars.DataFrame`, `pandas.DataFrame` or `numpy.ndarray` (one column). `columns` lists the columns the function reads, so only those are sent.

The worker that runs these functions (`python -m dataventura.worker`) and its protocol are described in [PROTOCOL.md](PROTOCOL.md).

> **Status:** pre-alpha. `@dv.transform` and the worker work; `@dv.view` and `@dv.reader` are planned. Not published yet; the first release is planned for the M3 milestone.
