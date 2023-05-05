# User
try:
    from geomulticorr.session import Open
    
# Developer
except ModuleNotFoundError:
   from src.geomulticorr.session import Open
