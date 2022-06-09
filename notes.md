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

# fish0_137
## Subject metadata
- Hatched: December 16, 2020
- Recording started: January 17, 2021 (32 days of age)
- Died: August 14, 2021 (241 days of age)

- Recordings start around 11:22 AM.

## Daylight savings time
Most files have on average 1.726M frames, which is consistent with a 20 Hz frame rate recorded 24 hrs per day minus 1 minute (20 Hz x 60 s/min x 60 min/hr x 24 hr/day ~= 1728000 frames/day).
- Recordings typically began: '11:22:13' of indicated day
- Recordings typically ended: '11:21:13' of next day

Prior to March 14, note that time stamps indicate that they started at 12:22...this is in STANDARD time (PST), and is equivalent to 11:22 PDT.
Daylight savings occured on March 14, 2021.

One day of recording, `p3_fish0_137_20210310.h5` seems like a clear concatenation of two days of recordings. It has 31488872 frames in a single file, which is not completely 2 days of recoridngs. The first timestamp is
- First timestamp:  2021-03-10 16:28:34.109630208
- Last timestamp:   2021-03-12 12:12:40.986110976

The number of frames and the start/end times of recordings surrounding that day are noted below:

| fileid | # frames | start time    | end time      |
| ------ | -------- | ------------- | ------------- |
| '0308' | 1.72M    | 3/08 12:22:13 | 3/09 12:21:13 |
| '0309' | 1.38M    | 3/09 12:22:13 | 3/10 07:40:32 |
| '0310' | 3.14M    | 3/10 16:28:34 | 3/12 12:12:40 |
| '0312' | 1.72M    | 3/12 12:22:43 | 3/13 12:21:42 |
| '0313' | 1.72M    | 3/13 12:22:43 | 3/14 12:21:43 |
| '0314' | 1.28M    | 3/14 17:29:53 | 3/15 11:19:09 |
| '0315' | 1.72M    | 3/15 11:22:13 | 3/16 11:21:13 |

We will not make any changes to this file, and continue to just take the first 1.72M frames.

## Omit truncated data
10 days of recordings which have <1.7M frames -- these truncated recordings are due to system errors and not due to fish behaviors, so their omission (should) not bias the data.
