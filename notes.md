# Reading timestamp
To read `frame_timestamp` values (-7hr shifts time into PST):
```
pd.to_datetime(df['frame_timestamp'], unit='s') - pd.Timedelta('07:00:00')
```

# MEMORY REQUIREMENTS
1 day of data = (~1726247 frames x 15 PCs) @ float32 = 98MB

FB-algorithm has O(KT) memory complexity, where K = # states and T = # frames
- (T=1726247 frames) x 40 states = O(263 MB) per day
- 10 days of data = 987 MB data + 2GB calculations ~= 3GB required
- 5 days of data  = 493 MB data + 1GB calculations ~= 1.5GB required


# Human-readable sizes
Code snippet, recursive implementation.
```
def humanize(bytes, units=[' bytes','KB','MB','GB','TB', 'PB', 'EB']):
    """Return a human readable string representation of bytes.

    Source: https://stackoverflow.com/a/43750422
    """

    return str(bytes) + units[0] if bytes < 1024 else humanize(bytes>>10, units[1:])
```
