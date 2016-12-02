TaxiBJ: InFlow/OutFlow, Meteorology and Holidays at Beijing
===========================================================

The TaxiBJ datasets consist of the following **SIX** datasets: 
- BJ16_M32x32_T30_InOut.h5
- BJ15_M32x32_T30_InOut.h5
- BJ14_M32x32_T30_InOut.h5
- BJ13_M32x32_T30_InOut.h5
- BJ_Meteorology.h5
- BJ_Holiday.txt

where the first four files are Crowd Flows at Beijing from the year 2013 to 2016, `BJ_Meteorology.h5` is the Meteorological data, `BJ_Holiday.txt` includes the holidays (and adjacent weekends) of Beijing. 

## Flows of Crowds

### BJ[YEAR]_M32x32_T30_InOut.h5
- YEAR: one of {13, 14, 15, 16}
- M32x32: the Beijing city is divided into a 32 x 32 grid map
- T30: time interval is equal to 30 minites, which means that there are 48 time intervals in a day
- InOut: Inflow/Outflow are defined in the following paper [1]. 
[1] Junbo Zhang, Yu Zheng, Dekang Qi. Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction. In AAAI 2017. 

Each file is a 4D tensor of shape (number_of_time_intervals, 2, 32, 32)

### example
```python
from deepst.datasets import stat
stat('BJ16_M32x32_T30_InOut.h5')
```

##

## BJ_Holiday