"""Exchange sessions independent of missing Yahoo price observations."""
import exchange_calendars as xcals
import pandas as pd

EXCHANGES={1:'XTKS',2:'XHKG',3:'XKRX',4:'XSHG'}
def sessions():
    return {g:pd.DatetimeIndex(xcals.get_calendar(name,start='2000-01-01',end='2025-12-31').sessions).tz_localize(None) for g,name in EXCHANGES.items()}
