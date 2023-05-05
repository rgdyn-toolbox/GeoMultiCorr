# User
try:
    from geomulticorr.session import open
    
# Developer
except ModuleNotFoundError:
   from src.geomulticorr.session import open
